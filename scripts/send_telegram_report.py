"""
Gửi báo cáo lãi/lỗ shop qua Telegram.

    python -m scripts.send_telegram_report export_shopee.csv export_tiktok.csv products.csv [ads_daily.csv] --days 7
    python -m scripts.send_telegram_report data/samples/shop_lan/{export_shopee,export_tiktok,products,ads_daily}.csv --dry-run

Cần TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID trong .env (trừ khi --dry-run).
ponytail: chạy theo lịch bằng cron / Task Scheduler / GitHub Actions; chưa có scheduler trong app
vì Render free tier ngủ khi rảnh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.services.telegram_report import build_report, load_shop_data, send_telegram


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="file export sàn, bảng sản phẩm, file quảng cáo (csv/xlsx)")
    parser.add_argument("--days", type=int, default=7, help="số ngày của kỳ báo cáo (mặc định 7)")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in báo cáo, không gửi")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 không in được tiếng Việt

    orders, ads = load_shop_data(args.files)
    report = build_report(orders, ads, days=args.days)
    if args.dry_run:
        print(report)
        return 0
    settings = get_settings()
    send_telegram(report, settings.telegram_bot_token, settings.telegram_chat_id)
    print("Đã gửi báo cáo qua Telegram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
