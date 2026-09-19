# Evaluation

Harness end-to-end: `tests/eval_100.py` upload 6 dataset trong `data/samples/`, hỏi 100 câu
qua API thật, chấm câu trả lời với **ground truth tính bằng pandas tại chỗ** (không fit theo output model).

## Scoring

5 chiều. `overall` = tổng có trọng số; 2 bộ trọng số tuỳ câu có kiểm cứng được hay không:

| chiều | ý nghĩa | trọng số HARD | trọng số SOFT |
|---|---|---|---|
| **correctness** | số khớp GT: lệch ≤2%→1.0 · ≤10%→0.8 · ≤30%→0.3 · còn lại 0.0. keyword khớp đủ→1.0 | **0.40** | — (n/a) |
| insight | có "so what" / khuyến nghị, không chỉ đọc số | 0.22 | 0.42 |
| no_meta | không leak chain-of-thought / meta-commentary | 0.16 | 0.28 |
| vn_natural | tiếng Việt tự nhiên, không dump JSON/markdown/bảng thô | 0.12 | 0.17 |
| concise | 2–5 câu, không cụt, không lan man | 0.10 | 0.13 |

- **HARD**: câu có ground-truth đơn trị (54/100). **SOFT**: câu mở / off_topic / bot_info (46/100) — `correctness` không áp dụng.
- `correctness` là chiều **cứng**, đối chiếu số bằng pandas — bộ số parser xử lý cả định dạng
  Việt ("8.950", "0,212") lẫn viết tắt "28,63 triệu" / "1,5 tỷ".
- 3 chiều mềm (`insight`/`no_meta`/`concise`/`vn_natural`) là **heuristic regex** → đã calibrate bằng 2 cách độc lập, xem bên dưới.
- `overall ≥ 0.70` = pass.

## Kết quả baseline (run 2026-09-12, quota sạch)

Gemini `flash-lite → flash`, không bị rate-limit storm (khác run 2026-09-10 trước đó — xem lịch sử bên dưới).
Chi tiết: [`eval-baseline/summary.json`](eval-baseline/summary.json) · [`eval-baseline/results.csv`](eval-baseline/results.csv).

| | |
|---|---|
| Pass | **84 / 100** |
| Avg overall | **0.870** |
| Avg correctness | **0.770** (54 câu kiểm cứng) |
| Avg insight / no_meta / concise / vn_natural | 0.82 / 0.96 / 0.91 / 0.97 |
| Fallback rule-based | **0 / 100** |
| HTTP 5xx | 0 |
| Latency | median **4.8 s** · mean **6.1 s** (mean ≈ median → không còn nhiễu 429 như run trước) |

### Theo dataset

| dataset | câu | pass | avg overall |
|---|---|---|---|
| sales_data | 27 | 25/27 | 0.93 |
| expense_claims | 11 | 11/11 | 0.91 |
| viet_relational | 6 | 5/6 | 0.88 |
| sales_sample | 24 | 21/24 | 0.88 |
| general_ledger | 16 | 12/16 | 0.82 |
| **financial_sample** | 16 | **9/16** | **0.75** |

### Theo nhóm câu hỏi

| nhóm | câu | pass | avg |
|---|---|---|---|
| bot_info | 5 | 5/5 | 0.94 |
| trend | 4 | 4/4 | 0.91 |
| comparison | 4 | 3/4 | 0.89 |
| ranking | 26 | 22/26 | 0.87 |
| aggregation | 39 | 31/39 | 0.86 |
| edge | 12 | 11/12 | 0.90 |
| language (câu hỏi kiểu chat) | 5 | 4/5 | 0.89 |
| off_topic | 5 | 4/5 | 0.71 |

## Calibration — heuristic có đáng tin không?

Hai nguồn độc lập, cùng kết luận: **`insight` là chiều yếu nhất**, `concise`/`vn_natural` tin được.

**1. Nhãn tay** (`tests/eval_calibration.py`, 30 câu trả lời thật đóng băng ở
[`eval-baseline/calibration-run-20260910.csv`](eval-baseline/calibration-run-20260910.csv), 1 người review):

| chiều | MAE | agreement (Δ≤0.25) | Pearson r |
|---|---|---|---|
| vn_natural | 0.03 | 100% | **+0.93** |
| concise | 0.12 | 83% | **+0.81** |
| **insight** | 0.28 | 60% | **+0.14** |

**2. LLM-as-judge trên full 100 câu** (`--judge`, baseline hiện tại, dùng chính failover chain của backend):

| chiều | n | MAE | agreement (Δ≤0.25) | Pearson r |
|---|---|---|---|---|
| concise | 87 | 0.079 | 87% | **+0.88** |
| vn_natural | 87 | 0.056 | 92% | **+0.73** |
| **insight** | 89 | 0.173 | 58% | **+0.33** |

(11–13 câu mỗi lần bị 429/parse-fail — free tier, không phải bug judge.)

→ Cả nhãn tay (n=30, r=0.14) lẫn LLM-judge (n=89, r=0.33) đều xác nhận: **quy tắc regex đếm
từ khoá cho `insight` không bám sát chất lượng thật**. `concise`/`vn_natural` thì tin được sau
khi vá lỗi truncation + template-dump (commit trước). Kết luận thực dụng: dùng `--judge` cho
`insight` khi cần con số đáng tin, giữ heuristic cho 3 chiều còn lại (rẻ, đủ tốt).

⚠️ Bộ nhãn tay 30 câu do 1 người review — đủ sanity-check, không phải human-eval quy mô lớn.

## Sửa cụm bug số liệu (2026-09-19)

Cụm bug bên dưới từng được ghi là "cột có dấu cách thừa → planner không resolve".
Đào kỹ thì **nguyên nhân thật khác hẳn**: cột tiền trong `financial_sample.csv`
là **chuỗi**, không phải số —

```
" $ 16,185.00 "      " $ (4,533.75) "      " $ -   "
```

`pd.to_numeric` trả `NaN` cho cả ba, `sum()` trên cột toàn `NaN` trả về `0`, và
agent tự tin báo *"Tổng lợi nhuận của toàn bộ dữ liệu là 0"*. Tên cột có khoảng
trắng chỉ là triệu chứng đi kèm, không phải nguyên nhân.

### Ground truth của chính bộ eval cũng sai

Hàm `_clean` trong `eval_100.py` chỉ bỏ `$` và `,`, nên `(4,533.75)` → `(4533.75)`
→ `NaN`. Nó **bỏ sót 63/700 dòng** cột `Profit` (58 số âm kế toán + 5 ô gạch ngang),
lệch **777.321,25**:

| | Tổng Profit |
|---|---|
| ground truth cũ (bỏ 63 dòng) | 17.671.023,54 |
| **đúng** | **16.893.702,29** |

Cả agent lẫn ground truth nay dùng chung `services/numeric_parse.py`. Nếu không,
hai bên sẽ tính trên hai tập số khác nhau và eval mất ý nghĩa.

### Ba bản sửa

| Bản sửa | Nội dung |
|---|---|
| `numeric_parse.py` | Đọc `$ (4,533.75)` → −4533.75, `1.234,56` (châu Âu), `32.000.000 VNĐ`, `$ -` → 0. Trả `None` (không phải 0) cho chữ, nếu không một cột tên sản phẩm sẽ thành cột toàn 0. Ngưỡng 90% mới ép kiểu cả cột. |
| `_repair_whole_dataset_aggregate` | "Tổng Debit của **toàn bộ** sổ cái" hay bị dịch thành `group_by=["AccountName"] + limit 1` → agent trả tổng của **một** tài khoản mà vẫn gọi là "toàn bộ". Bỏ `group_by` khi câu hỏi nhắm cả tập và không có dấu hiệu chia nhóm. |
| `_repair_id_to_name_group` | Nhóm theo `product_id` cho ra "P002"; người đọc cần "MacBook Air M2". Đổi sang cột tên khi bảng có cả hai, trừ khi câu hỏi hỏi đúng mã. |

Chuẩn hoá chạy một lần tại `SessionStore._normalize_frame()` — điểm mà cả 7 đường
nạp dữ liệu đều đi qua.

### Kết quả trên 12 câu từng fail

**Đã sửa (8):** `31, 33, 34, 36, 43` (financial_sample) · `46, 47` (general_ledger) · `72`

Ví dụ `[31]`: từ *"Tổng lợi nhuận là 0"* → **16.893.702,29**, khớp ground truth.
`[46]`: từ *"397.936,33, ghi nhận dưới tài khoản COGS"* → **1.210.240,47**.

**Còn lại (4), mỗi câu một loại vấn đề khác:**

| # | Hiện trạng | Bản chất |
|---|---|---|
| `38` | agent trả "Tháng 10", GT `" October "` | **Agent đúng, chấm điểm sai** — so khớp từ khoá không hiểu October = tháng 10 |
| `48` | net balance ra sai | Grammar plan **không có phép trừ giữa hai metric**; cần thêm derived op |
| `60` | `concise`/`vn_natural` thấp | Rơi vào định dạng "Executive Brief" của nhánh tổng hợp dự phòng |
| `71` | **dao động**: lúc trả 12 (đúng), lúc 32 | LLM chọn sai sheet trong file nhiều sheet — không ổn định giữa các lần chạy |

Bốn câu này **không** sửa được bằng cùng một cách với cụm trên: một câu là lỗi
của thang điểm, một câu là giới hạn của grammar, một câu là định dạng đầu ra,
một câu là tính bất định của LLM.

## Bug agent do eval phát hiện (chưa sửa — ngoài scope harness)

Ổn định qua 2 lần chạy độc lập (2026-09-10 và 2026-09-12) — không phải nhiễu ngẫu nhiên:

| triệu chứng | dataset | ví dụ |
|---|---|---|
| Cột có dấu cách thừa (`" Profit "`, `"  Sales "`) → planner không resolve, trả 0 / "không tìm thấy" | financial_sample | #31 (GT 17.6M, trả 0), #33, #34, #36, #38, #43, #44 |
| Sổ cái đa tiền tệ → cộng gộp AUD+CAD+EUR+GBP+USD, số vô nghĩa | general_ledger | #46, #47, #48, #60 |
| Multi-sheet → đếm nhầm bảng | viet_relational | #71 (GT 12 khách hàng) |
| Sai logic nhóm (region/segment) dù không liên quan cột lỗi | sales_sample | #9 (model trả "North", GT thật là "East") |

## Ghi chú độ tin cậy

- Run 2026-09-12 quota sạch — mean latency ≈ median, không còn retry-storm 429 như run
  2026-09-10 trước đó (khi đó mean bị kéo lên 75s do vài request retry 15–17 phút).
- 1 câu (#88) client read-timeout đơn lẻ (60s) — hạ tầng, không phải lỗi logic.
- **Biến động giữa 2 lần chạy cùng bộ câu hỏi**: category `language` đổi từ 1/5 (run trước) lên
  4/5 (run này) — do `temperature=0.3` ở bước synthesize. Baseline 1 lần chạy không đại diện
  tuyệt đối; số ổn định qua nhiều lần chạy là cụm bug ở financial_sample/general_ledger.

## Lần chạy 2026-09-19 (chuỗi 2 key, có đầy đủ chỉ số mới)

`docs/eval-baseline/summary-20260919.json` — baseline cũ (`summary.json`, 2026-09-12)
được giữ nguyên để so.

| | 2026-09-12 | 2026-09-19 | |
|---|---|---|---|
| pass | 84/100 | **86/100** | ▲ +2 |
| avg_overall | 0.870 | 0.895 | ▲ +0.025 |
| avg_correctness | 0.770 | 0.789 | ▲ +0.019 |
| avg_insight | 0.816 | 0.856 | ▲ +0.040 |
| median latency | 4.838 ms | **2.508 ms** | ▼ −48% |
| p95 / p99 latency | — | 3.205 / 4.061 ms | mới |
| tokens | — | 256.623 (2.566/câu) | mới |
| LLM calls | — | 187 (1,87/câu) | mới |
| provider errors | — | 4 × `ResourceExhausted` (đã failover) | mới |
| fallback về rule-based | — | 0/100 | mới |

**Đã fix:** `[9, 44, 81, 88]` · **Fail mới:** `[72, 77]`

### ⚠️ Không quy toàn bộ cải thiện cho một nguyên nhân

Giữa hai lần chạy có **nhiều biến cùng thay đổi**, nên +2 điểm pass không phải
bằng chứng sạch cho bất kỳ thay đổi riêng lẻ nào:

- định tuyến intent được viết lại (ảnh hưởng trực tiếp `[81]`, `[77]`);
- `sanitize_for_prompt` chèn vào schema line → **prompt planner đã khác đi**;
- chuỗi model thêm key phụ, và lần 09-12 chạy khi quota sạch còn lần này gặp 4 lần 429;
- `temperature=0.3` ở bước synthesize vẫn gây biến động giữa các lần chạy — như
  chính tài liệu này đã ghi nhận ở phần "Ghi chú độ tin cậy".

Riêng độ trễ giảm 48% gần như chắc chắn **không** đến từ code trong repo: không
có thay đổi nào rút ngắn đường xử lý. Nhiều khả năng do model/hạ tầng phía Gemini.

### Hai fail mới nói lên điều gì

**`[77]` "Hệ thống này có thể phân tích file Excel không?" → 0.681** (ngưỡng 0.7).
Định tuyến giờ **đúng** (`bot_info` thay vì `data_query`) và nội dung trả lời cũng
đúng, nhưng `concise` chỉ 0.7 vì câu trả lời vòng vo — mở đầu bằng mô tả file đã
upload rồi mới vào ý chính. Tức là sửa định tuyến đã **làm lộ ra** chất lượng
diễn đạt của `bot_info_node`, chứ không phải gây ra lỗi mới.

**`[72]` "Sản phẩm nào có giá bán cao nhất?" → correctness 0.0.** Lỗi số liệu
thuần, không liên quan định tuyến.

Đổi lại, `[81]` "Thủ đô của nước Pháp là đâu?" từ fail lên **1.00** nhờ định tuyến
`off_topic` hoạt động đúng.

### Cụm bug cũ vẫn còn nguyên

`financial_sample` (`31, 33, 34, 36, 38, 43`) và `general_ledger` (`46, 47, 48, 60`)
tiếp tục fail đúng như mô tả ở mục "Bug agent do eval phát hiện" — cột có dấu cách
thừa và sổ cái đa tiền tệ. Chưa đụng tới, vẫn nằm ngoài phạm vi harness.

## Đo lường vận hành (token · chi phí · đuôi độ trễ)

`backend/app/services/usage.py` đo từng lần gọi LLM và trả kèm trong response
API (`ChatResponse.usage`), nên `eval_100.py` gom được số mà không cần đoán.

Bổ sung vào summary: `p95_latency_ms`, `p99_latency_ms`, `max_latency_ms`,
`llm_calls`, `prompt_tokens`, `completion_tokens`, `cost_usd`, `provider_errors`.

**Vì sao thêm p95:** baseline trước đó chỉ có median 4.838ms và mean 6.074ms.
Mean cao hơn median cho biết phân phối lệch phải, nhưng không nói được đuôi dài
tới đâu — mà đuôi mới là cái người dùng thật sự cảm thấy.

**Chi phí là số suy ra, không phải số đo.** Token do provider trả về nên đo được;
chi phí phải quy từ bảng giá khai trong `LLM_PRICING_JSON`. Khi model chưa khai
giá, summary in `n/a` kèm số `unpriced_calls` thay vì im lặng báo `$0` — một
báo cáo chi phí $0 vì thiếu cấu hình còn tệ hơn không có báo cáo.

### Cái mà đo lường này phát hiện ra ngay lần chạy đầu

Một câu hỏi thật (`tổng doanh thu theo vùng`, 2026-09-19):

```json
{"llm_calls": 8, "llm_calls_failed": 6, "total_tokens": 2868,
 "failures_by_type": {"Unauthenticated": 6},
 "models_used": ["nvidia/nemotron-3-super-120b-a12b:free"]}
```

6/8 lần gọi hỏng vì `Unauthenticated` — toàn bộ nhánh Gemini chết, câu trả lời
đúng là nhờ failover tụt xuống OpenRouter. Chỉ số `llm_usage_rate: 100%` của
baseline cũ **không hề thấy chuyện này**: nó chỉ đếm "có rơi xuống rule-based
hay không", nên một chuỗi failover 6 lần hỏng vẫn được tính là 100% khỏe mạnh.

Đây đúng là lý do cần tách lỗi theo nguyên nhân (`failures_by_type`) thay vì
một tỉ lệ gộp.

## Định tuyến intent (tool-selection accuracy)

`tests/test_routing.py` — offline, `classify_query` thuần rule-based nên không tốn quota.

**Accuracy trần là chỉ số lừa dối ở đây.** 85/100 câu trong bộ eval là câu hỏi dữ
liệu, nên một bộ phân loại luôn trả `data_query` đạt ngay 85% mà không phân loại
được gì. Chỉ số chính phải là **macro-recall**.

| | Trước | Sau |
|---|---|---|
| Bộ eval — accuracy | 92.0% | 100% |
| Bộ eval — macro-recall | 64.0% | 100% |
| **Held-out — macro-recall** | **33.3%** | **100%** |
| Held-out `bot_info` | 0/8 | 8/8 |
| Held-out `off_topic` | 0/10 | 10/10 |

Con số 92% ban đầu che giấu việc `bot_info` chỉ đúng 1/5 và `off_topic` 1/5 — bộ
phân loại gần như luôn trả `data_query`, chỉ hơn baseline ngây thơ 7 điểm.

**Nguyên nhân:** khớp chuỗi con cứng. `"viết thơ"` không bắt được *"viết cho tôi
một bài thơ"*; `"nấu ăn"` không bắt được *"cách nấu món phở"*. Bản sửa thay bằng
**mẫu cấu trúc** (regex mô tả dạng câu: `<chủ thể là công cụ> + <động từ khả năng>`),
kèm cơ chế **tín hiệu dữ liệu phủ quyết** off-topic — để *"Nên mua thêm sản phẩm
nào dựa trên doanh số?"* không bị đẩy đi cùng *"Năm nay có nên mua vàng?"*.

⚠️ **Giới hạn của con số held-out.** Tập này được viết trước khi sửa, nhưng trong
lúc sửa danh sách câu sai của nó *đã* được in ra để phân tích. Held-out đúng nghĩa
thì không được nhìn, nên 100% ở đây lạc quan hơn thực tế. Cả hai tập cũng do cùng
một người viết. Nó đủ chứng minh bộ phân loại không còn chỉ khớp đúng 100 câu eval,
nhưng không đủ để tuyên bố một tỉ lệ chính xác ngoài đời thật.

Lỗi tốn kém nhất — câu hỏi dữ liệu bị coi là chit-chat — được chốt riêng bằng
`test_data_questions_never_misrouted` với ngưỡng recall 100% trên cả hai tập.

## Quy chi phí về từng node (cost attribution)

`usage.stage()` gán mỗi lần gọi LLM cho node phát sinh ra nó, nên `by_stage` trả
lời được "token tiêu **ở đâu**", không chỉ "tiêu bao nhiêu". Đo thật, một câu hỏi
(`tổng doanh thu theo vùng`, 2026-09-19):

| Node | Token | % | Latency |
|---|---|---|---|
| `plan` | 1.815 | **83%** | 1.325 ms |
| `synthesize` | 372 | 17% | 1.262 ms |

Planner chiếm 83% token vì prompt chứa nhiều ví dụ few-shot, nhưng độ trễ lại chia
gần đôi. Nghĩa là: muốn giảm **chi phí** thì rút gọn prompt planner; muốn giảm
**độ trễ** thì phải động tới cả hai bước.

Cách này thay cho một dịch vụ tracing ngoài (LangSmith/Langfuse): không thêm phụ
thuộc, chạy được trong test, và số liệu đi thẳng vào response API.

## Đánh giá đối kháng (security)

`tests/test_adversarial.py` — 18 đòn tấn công, chạy offline (không LLM, không server,
không chạm internet; SSRF kiểm bằng HTTP server dựng tạm trên loopback).

| Nhóm tấn công | Chặn | Tổng | Tỷ lệ |
|---|---|---|---|
| SSRF | 7 | 7 | 100% |
| Plan escape | 6 | 6 | 100% |
| Prompt injection | 5 | 5 | 100% |
| **Tổng** | **18** | **18** | **100%** |

⚠️ 100% ở đây nghĩa là **mọi vector đã biết trong bộ này đều bị chặn** — bộ tấn công do
chính tác giả thiết kế, nên con số này không đồng nghĩa "hệ thống an toàn". Nó đo được
việc các lỗ hổng đã phát hiện không tái phát.

### Lỗ hổng phát hiện được và đã vá

| # | Lỗ hổng | Mức | Bằng chứng trước khi vá |
|---|---|---|---|
| 1 | **SSRF** — `fetch_from_url` nhận URL người dùng dán, không lọc scheme, không chặn IP nội bộ, đi theo redirect vô điều kiện, không giới hạn kích thước tải | Cao | PoC kéo được nội dung từ dịch vụ chạy trên `127.0.0.1`. Repo deploy lên Railway/Render → `169.254.169.254` làm lộ credentials |
| 2 | **ReDoS** — filter `contains` gọi `str.contains` để mặc định `regex=True`, giá trị filter do LLM sinh (chịu ảnh hưởng câu hỏi người dùng) | Trung bình–Cao | Pattern `(a+)+$`: thời gian ×4 mỗi 2 ký tự thêm vào (len=22 → 0.64s/dòng). 50 dòng treo >120s |
| 3 | **Indirect prompt injection** — tên cột và 5 giá trị mẫu mỗi cột nhúng thẳng vào prompt planner, không sanitize | Trung bình | Tác hại bị giới hạn bởi thiết kế plan-grammar: injection thành công cũng chỉ sinh được plan JSON hợp lệ, không dẫn tới thực thi code |
| 4 | **limit không có trần** | Thấp | `limit: 10**9` qua được validate, phình payload và token khi synthesize |

Bản vá nằm ở `backend/app/services/security.py` (`safe_fetch`, `assert_public_url`,
`sanitize_for_prompt`, `LITERAL_CONTAINS`).

**Giới hạn còn lại — nói rõ để không phóng đại:**
- `assert_public_url` phân giải DNS rồi mới request, nên về lý thuyết vẫn còn khe hở
  DNS rebinding. Bịt hẳn phải connect thẳng bằng IP đã validate và tự set header `Host`.
- `sanitize_for_prompt` lọc theo danh sách cụm từ — chặn được vector rẻ tiền, không
  chặn được payload viết lại khéo. Lớp phòng thủ thật vẫn là plan-grammar + validate.

## Chạy lại

```bash
# bộ đối kháng — offline, dùng trong CI
pytest tests/test_adversarial.py -q
python tests/test_adversarial.py          # in bảng block rate

# eval đầy đủ (cần server + GEMINI_API_KEY)
uvicorn backend.app.main:app --port 8000 &
python tests/eval_100.py --base-url http://localhost:8000 --delay 2 --judge \
    --baseline docs/eval-baseline/summary.json      # kèm bảng regression + judge agreement

# chấm lại run cũ bằng scoring hiện tại — KHÔNG cần server, KHÔNG tốn quota
python tests/eval_100.py --rescore docs/eval-baseline/results.csv

# calibration nhãn tay (trên bộ câu trả lời đóng băng 2026-09-10)
python tests/eval_calibration.py
```
