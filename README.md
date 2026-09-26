# SellerLens

**English** · [Tiếng Việt](README.vi.md)

[![CI](https://github.com/luongducthangDS/data_analysis_ai_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/luongducthangDS/data_analysis_ai_agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Ask your marketplace exports why profit dropped. Get numbers you can check, not numbers an LLM made up.**

Multi-channel sellers (Shopee, TikTok Shop) have order exports but rarely know their **real** profit after platform fees, cost of goods, returns and ads. SellerLens takes the raw exports plus a `sku, cost` table and answers questions in plain language (Vietnamese UI):

- *What's my real profit by SKU / channel / month?*
- *What % of revenue do platform fees eat?*
- *Why did profit drop in May, and what should I check first?*

<!-- TODO: 20s demo GIF — upload export → "why did profit drop in May?" → profit bridge + action list -->

**Live demo:** https://data-analysis-ai-agent-tb47.onrender.com (free tier, first load can take ~1 min to wake up). Upload [`data/samples/shop_lan/shop_lan_orders.csv`](data/samples/shop_lan/) and ask *"vì sao lãi tháng 5 giảm?"* ("why did profit drop in May?").

## How it's different

Most "chat with your data" tools let the LLM write SQL or Python. That means arbitrary code execution, and wrong numbers that look right.

1. **The LLM never writes code.** It only emits a small JSON plan (`aggregate`, `group_by`, `metrics`, `filter`…). The backend validates it against the real DataFrame schema, then runs it with plain pandas. No `eval`, no `exec`, no SQL. If the LLM fails, a rule-based planner takes over.
2. **Money metrics are defined once, in code.** Net revenue, platform fees and profit are computed deterministically at load time ([`ecommerce_semantic.py`](backend/app/services/ecommerce_semantic.py)). The LLM can only sum them.
3. **Answers are grounded.** A summary is rejected if it contains a number that isn't in the computed result.
4. **Missing data is refused, not guessed.** No cost table → profit questions get a fixed "can't compute profit yet" answer plus a pre-filled cost template, instead of revenue labelled as profit. Costs missing for some SKUs → every profit answer carries a warning with the share of revenue affected. Measured by [`tests/eval_seller.py`](tests/eval_seller.py): 25 questions × 3 upload modes (full / no costs / 1/3 costs missing), scored against the simulated ground truth — last run 25/25, **0 wrong-but-plausible numbers**.

Example plan:

```json
{"action":"aggregate","group_by":["kenh"],
 "metrics":[{"column":"loi_nhuan_truoc_qc","aggregation":"sum","label":"Profit"}],
 "sort":[{"column":"Profit","direction":"desc"}],"limit":10}
```

### "Why did profit change?"

The `profit_bridge` action splits the profit delta between two periods into revenue, platform fees, COGS and returns. The parts **add up exactly** to the delta. It isolates the part caused by a higher *fee rate*, points to the channel / SKU / province × carrier dragging profit down, and emits a to-check list from fixed rules (not LLM ideas). For example: *"fee rate 33.5% → 36.5% cut profit by 13.3M VND; raise prices ~4.7% to keep take-home"*.

## Quickstart

### Docker (one command)

```bash
cp .env.example .env        # set GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
docker compose up --build
# → http://localhost:8000
```

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                   # app + pytest/ruff
cp .env.example .env                                  # set GEMINI_API_KEY
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app --port 8000              # UI: :8000 · API docs: :8000/docs
```

Optional: `GEMINI_API_KEY2` doubles free-tier quota, `OPENROUTER_API_KEY` adds a fallback provider chain. See [`.env.example`](.env.example).

## Demo data

[`data/samples/shop_lan/`](data/samples/shop_lan/) is **simulated** ("Ms. Lan's shop": 12,000 orders, 6 months). Fee parameters come from press coverage of marketplace fee schedules; the Shopee/TikTok order split (64/36) follows Q1/2025 market share from Metric.vn. **It is not a real shop's data.** It ships with 4 seeded scenarios and their answers (`ground_truth.json`): a fee hike from May, a best-selling SKU that loses money, a TikTok ad spike in week 23, and rising returns in one province × carrier.

## Other features

The core still reads any CSV/Excel file:

- Upload CSV / XLSX / XLS (multiple files), or paste a public CSV / Google Sheets / Dropbox URL
- Auto data profile, streaming chat with inline Plotly charts, auto-generated dashboard
- Multi-sheet: detects relationships, cross-sheet joins, warns on 1-to-many fan-out
- Export processed data (CSV) and reports (Markdown)
- Bring your own key (Gemini / Anthropic), stored in the browser only

## Evaluation

Measured per dimension, not rolled into one score. Details: [`docs/EVALUATION.md`](docs/EVALUATION.md).

| Dimension | Result |
|---|---|
| Numeric correctness | 0.79 avg on 53 single-answer questions, vs. pandas ground truth |
| Adversarial (SSRF, plan escape, prompt injection) | 18/18 blocked, runs offline in CI |
| Latency | median 2.5 s · p95 3.2 s (dev machine, free tier) |

The question set was written by the author over 6 sample datasets, not by real shop owners.

## Limitations

- MVP, not yet used by a real shop.
- Data is sent to third-party LLMs (Google, OpenRouter). No self-hosted model option yet.
- In-memory pandas: files must fit in RAM (tens of MB is fine for a small shop).
- Ads spend is allocated to orders by net revenue within the same day × channel (`loi_nhuan_rong`). Net profit per month/channel is exact; per SKU/province it is an estimate.
- Marketplace exports don't include damaged-return write-offs, so profit from exports is higher than the simulated ground truth by exactly that amount (tested).
- Basic auth only (`X-API-Key`), no multi-tenant.

## Docs

[Architecture](ARCHITECTURE.md) · [Engineering details](docs/ENGINEERING.md) (in Vietnamese) · [Evaluation](docs/EVALUATION.md)

**Stack:** FastAPI · LangGraph · pandas · Plotly · React/Vite · Gemini + OpenRouter · Docker

## Contributing

Issues and PRs welcome. Adding a new marketplace export format is a good first contribution: it's one dict in `EXPORT_HEADERS` ([`ecommerce_semantic.py`](backend/app/services/ecommerce_semantic.py)).

```bash
pytest -q
```

## License

[MIT](LICENSE) © 2026 Luong Duc Thang
