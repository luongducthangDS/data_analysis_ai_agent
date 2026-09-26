"""Lớp phòng thủ cho các ranh giới tin cậy (trust boundary).

Ba đầu vào không tin cậy đi vào hệ thống:
  1. URL người dùng dán  → `safe_fetch()`        chống SSRF
  2. Giá trị filter từ LLM plan → `LITERAL_CONTAINS`  chống ReDoS
  3. Nội dung file upload đi vào prompt → `sanitize_for_prompt()`  giảm prompt injection

"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------- SSRF

ALLOWED_SCHEMES = {"http", "https"}
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024   # khớp giới hạn 10MB của upload file
MAX_REDIRECTS = 5
FETCH_TIMEOUT = 30


class BlockedURLError(ValueError):
    """URL trỏ tới tài nguyên nội bộ hoặc vi phạm chính sách tải."""


def _ip_is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    # is_global loại sẵn loopback/private/link-local/reserved/multicast,
    # bao gồm 169.254.169.254 (cloud metadata) và ::1 / fc00::/7.
    return addr.is_global and not addr.is_multicast


def assert_public_url(url: str) -> None:
    """Chặn URL trỏ vào mạng nội bộ. Ném BlockedURLError nếu không an toàn."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise BlockedURLError(
            f"Chỉ hỗ trợ http/https, không chấp nhận scheme {parsed.scheme!r}."
        )
    host = parsed.hostname
    if not host:
        raise BlockedURLError("URL thiếu hostname.")

    try:
        infos = socket.getaddrinfo(host, parsed.port or 0, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedURLError(f"Không phân giải được hostname {host!r}.") from exc

    # Mọi IP mà hostname phân giải ra đều phải public — chặn cả bản ghi DNS
    # cố tình trỏ về nội bộ (vd. một domain public trỏ A record 127.0.0.1).
    for info in infos:
        ip = info[4][0]
        if not _ip_is_public(ip):
            raise BlockedURLError(
                f"URL trỏ tới địa chỉ nội bộ ({ip}) — bị chặn để phòng SSRF."
            )


def safe_fetch(url: str, *, timeout: int = FETCH_TIMEOUT) -> requests.Response:
    """GET một URL công khai, tự đi theo redirect và kiểm tra lại từng chặng.

    `requests` với allow_redirects=True sẽ bỏ qua mọi kiểm tra ở các hop sau,
    nên ở đây tự lần theo redirect và validate lại mỗi lần.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        assert_public_url(current)
        resp = requests.get(
            current, timeout=timeout, allow_redirects=False, stream=True
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("location", "")
            resp.close()
            if not location:
                raise BlockedURLError("Redirect thiếu header Location.")
            current = requests.compat.urljoin(current, location)
            continue
        return _read_capped(resp)
    raise BlockedURLError(f"Vượt quá {MAX_REDIRECTS} lần redirect.")


def _read_capped(resp: requests.Response) -> requests.Response:
    """Đọc body nhưng cắt ở MAX_DOWNLOAD_BYTES.

    Content-Length có thể nói dối hoặc vắng mặt, nên vẫn phải đếm khi đọc.
    """
    declared = resp.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
        resp.close()
        raise BlockedURLError(
            f"File vượt giới hạn {MAX_DOWNLOAD_BYTES // 1024 // 1024}MB."
        )

    chunks, total = [], 0
    for chunk in resp.iter_content(64 * 1024):
        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            resp.close()
            raise BlockedURLError(
                f"File vượt giới hạn {MAX_DOWNLOAD_BYTES // 1024 // 1024}MB."
            )
        chunks.append(chunk)
    resp._content = b"".join(chunks)      # noqa: SLF001 — nạp lại body đã cắt
    resp._content_consumed = True         # noqa: SLF001
    return resp


# ---------------------------------------------------------------- ReDoS

# `pandas.Series.str.contains` mặc định regex=True. Giá trị filter đến từ LLM
# plan (chịu ảnh hưởng của câu hỏi người dùng), nên một pattern như "(a+)+$"
# gây catastrophic backtracking — đo được: mỗi 2 ký tự làm thời gian ×4.
# Ý định của `contains` ở đây luôn là so khớp chuỗi con, không phải regex.
LITERAL_CONTAINS = False   # truyền thẳng vào tham số `regex=` của pandas


# ---------------------------------------------------------------- Prompt injection

MAX_PROMPT_CELL_LEN = 120

_INJECTION_MARKERS = (
    "bỏ qua", "ignore previous", "ignore the above", "disregard",
    "system prompt", "quy tắc tuyệt đối", "new instruction",
    "you are now", "bạn là", "### instruction", "<|im_start|>",
)


def sanitize_for_prompt(value: object) -> str:
    """Làm sạch một giá trị ô / tên cột trước khi nhúng vào prompt LLM.

    Không thể loại bỏ hoàn toàn indirect prompt injection, nhưng cắt được
    các vector rẻ tiền: xuống dòng giả lập cấu trúc prompt, chuỗi quá dài,
    và các cụm ra lệnh phổ biến.
    """
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.replace("`", "'")
    if len(text) > MAX_PROMPT_CELL_LEN:
        text = text[:MAX_PROMPT_CELL_LEN] + "…"
    lowered = text.lower()
    for marker in _INJECTION_MARKERS:
        if marker in lowered:
            return "[nội dung bị lọc vì giống chỉ thị]"
    return text


def looks_like_injection(value: object) -> bool:
    """True nếu giá trị chứa dấu hiệu ra lệnh — dùng cho đo lường/eval."""
    lowered = str(value).lower()
    return any(marker in lowered for marker in _INJECTION_MARKERS)
