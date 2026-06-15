from __future__ import annotations

import io
import json
import os
import re

import pandas as pd
import requests


def fetch_from_url(url: str) -> tuple[str, bytes]:
    """Download file từ URL public. Tự detect Google Sheets URL → export CSV."""
    url = _normalize_gsheet_url(url)
    try:
        resp = requests.get(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("Request timeout sau 30 giây.")
    except requests.exceptions.HTTPError as exc:
        raise ValueError(f"HTTP {exc.response.status_code}: {exc.response.reason}") from exc
    filename = _infer_filename(url, resp)
    return filename, resp.content


def fetch_from_gsheet(url_or_id: str, sheet_name: str | None = None) -> tuple[str, bytes]:
    """Đọc Google Sheet qua service account → (filename, xlsx_bytes)."""
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        raise ValueError(
            "GOOGLE_CREDENTIALS_JSON chưa được cấu hình. "
            "Admin cần set biến môi trường này với nội dung file service account JSON."
        )

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise RuntimeError("Thiếu package. Chạy: pip install gspread google-auth") from exc

    try:
        creds_data = json.loads(creds_json)
    except json.JSONDecodeError as exc:
        raise ValueError("GOOGLE_CREDENTIALS_JSON không phải JSON hợp lệ.") from exc

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(creds_data, scopes=scopes)
    gc = gspread.authorize(creds)

    sheet_id = _extract_sheet_id(url_or_id)
    try:
        spreadsheet = gc.open_by_key(sheet_id)
    except gspread.exceptions.SpreadsheetNotFound:
        raise ValueError(
            f"Không tìm thấy spreadsheet '{sheet_id}'. "
            "Kiểm tra Sheet đã được chia sẻ với email service account chưa."
        )

    if sheet_name:
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            available = [ws.title for ws in spreadsheet.worksheets()]
            raise ValueError(f"Sheet '{sheet_name}' không tồn tại. Có: {available}")
    else:
        worksheet = spreadsheet.sheet1

    records = worksheet.get_all_records()
    if not records:
        raise ValueError("Sheet không có dữ liệu (hoặc không có header row).")

    df = pd.DataFrame(records)
    filename = f"{spreadsheet.title}.xlsx"
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return filename, buf.getvalue()


def _normalize_gsheet_url(url: str) -> str:
    """Chuyển Google Sheets /edit hoặc /pub URL → export CSV URL."""
    match = re.search(r"spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        return url  # Không phải Google Sheets — giữ nguyên
    sheet_id = match.group(1)
    gid_match = re.search(r"[#&?]gid=(\d+)", url)
    gid = f"&gid={gid_match.group(1)}" if gid_match else ""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid}"


def _extract_sheet_id(url_or_id: str) -> str:
    match = re.search(r"spreadsheets/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    # Raw ID: chỉ chứa base64url chars, ít nhất 20 ký tự
    if re.match(r"^[a-zA-Z0-9_-]{20,}$", url_or_id.strip()):
        return url_or_id.strip()
    raise ValueError(
        f"Không thể xác định Google Sheet ID từ: '{url_or_id}'. "
        "Hãy dán URL đầy đủ từ thanh địa chỉ trình duyệt."
    )


def _infer_filename(url: str, resp: requests.Response) -> str:
    """Suy ra tên file từ Content-Disposition header hoặc URL path."""
    cd = resp.headers.get("content-disposition", "")
    match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)', cd)
    if match:
        return match.group(1).strip().strip('"\'')
    # Thử lấy từ URL path
    path = url.split("?")[0].rstrip("/").split("/")[-1]
    if "." in path and len(path) < 100:
        return path
    # Suy từ Content-Type
    ct = resp.headers.get("content-type", "")
    if "csv" in ct:
        return "import.csv"
    if "spreadsheet" in ct or "excel" in ct or "openxmlformats" in ct:
        return "import.xlsx"
    return "import.csv"
