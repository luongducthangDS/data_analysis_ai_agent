from __future__ import annotations

import re

import requests

from backend.app.services.security import BlockedURLError, safe_fetch


def fetch_from_url(url: str) -> tuple[str, bytes]:
    """Download file từ URL public. Tự detect Google Sheets URL → export CSV."""
    url = _normalize_gsheet_url(url)
    try:
        # safe_fetch: chặn URL trỏ vào mạng nội bộ (kể cả qua redirect) và
        # cắt body ở 10MB. Xem services/security.py.
        resp = safe_fetch(url, timeout=30)
        resp.raise_for_status()
    except BlockedURLError as exc:
        raise ValueError(str(exc)) from exc
    except requests.exceptions.Timeout:
        raise ValueError("Request timeout sau 30 giây.")
    except requests.exceptions.HTTPError as exc:
        raise ValueError(f"HTTP {exc.response.status_code}: {exc.response.reason}") from exc
    filename = _infer_filename(url, resp)
    return filename, resp.content


def _normalize_gsheet_url(url: str) -> str:
    """Chuyển Google Sheets /edit hoặc /pub URL → export CSV URL."""
    match = re.search(r"spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        return url  # Không phải Google Sheets — giữ nguyên
    sheet_id = match.group(1)
    gid_match = re.search(r"[#&?]gid=(\d+)", url)
    gid = f"&gid={gid_match.group(1)}" if gid_match else ""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid}"


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
