"""
Query intent classifier.

Classifies user questions into:
  - "bot_info"      → questions about the chatbot itself
  - "off_topic"     → questions unrelated to data analysis
  - "data_summary"  → requests for dataset overview/description (routes to profile node)
  - "data_query"    → specific analytical questions about the dataset (default)
"""
from __future__ import annotations

import re

_BOT_INFO: list[str] = [
    # greetings → respond as bot introduction
    "chào bạn", "xin chào", "chào em", "chào anh", "chào chị",
    "chào buổi sáng", "chào buổi tối", "kính chào",
    "hello", "hi there", "hey there", "good morning", "good afternoon", "good evening",
    # identity
    "bạn là ai", "bạn tên gì", "bạn là gì",
    # capability — sub-phrases that catch all variants (có thể / làm được / giúp được)
    "làm được gì", "có thể làm gì", "có thể giúp gì", "giúp được gì",
    "bạn hỗ trợ gì", "hỗ trợ những gì", "có khả năng gì", "chức năng gì",
    "tính năng của bạn", "bạn có tính năng gì",
    "who are you", "what are you", "what can you do", "what do you do",
    "tell me about yourself", "your capabilities", "your features",
    "hướng dẫn sử dụng", "cách dùng", "cách sử dụng",
    "how to use", "how do i use",
    "bạn được tạo bởi", "bạn do ai tạo", "who made you", "who created you",
    "xây dựng bởi", "developed by", "built by",
    "bạn là chatbot", "bạn là ai vậy", "em là ai",
    # help requests — conversational, không liên quan data
    "cần bạn hỗ trợ", "cần hỗ trợ", "cần giúp đỡ", "cần giúp",
    "nhờ bạn giúp", "giúp tôi với", "giúp mình với",
    "need help", "help me", "can you help", "i need help",
    "bạn có thể giúp tôi", "bạn giúp tôi",
]

# Short standalone greetings (≤3 words) that contain these tokens
_GREETING_TOKENS: list[str] = ["chào", "hello", "hi", "hey", "alo"]

_DATA_SUMMARY: list[str] = [
    # Vietnamese — diacritics
    "tóm tắt", "mô tả dữ liệu", "tổng quan", "hiểu dữ liệu",
    "muốn hiểu", "muốn biết về dữ liệu", "dữ liệu có gì",
    "bao nhiêu cột", "bao nhiêu dòng", "bao nhiêu hàng",
    "thống kê mô tả", "khám phá dữ liệu", "phân tích tổng quan",
    "phân phối dữ liệu", "cấu trúc dữ liệu", "các cột", "loại dữ liệu",
    "cho tôi biết về dữ liệu", "cho biết về dữ liệu", "giải thích dữ liệu",
    # Vietnamese — no diacritics (normalized)
    "tom tat", "mo ta du lieu", "tong quan", "hieu du lieu",
    "muon hieu", "muon biet ve du lieu", "du lieu co gi",
    "bao nhieu cot", "bao nhieu dong", "bao nhieu hang",
    "thong ke mo ta", "kham pha du lieu", "phan tich tong quan",
    "phan phoi du lieu", "cau truc du lieu", "loai du lieu",
    # English
    "describe", "overview", "summary", "summarize",
    "what columns", "how many rows", "how many columns",
    "data types", "column types", "explore the data",
    "understand the data", "tell me about the data",
    "what is in the data", "dataset overview",
]

_OFF_TOPIC: list[str] = [
    "thời tiết", "weather", "nhiệt độ hôm nay", "dự báo thời tiết",
    "tin tức", "news", "thời sự", "bản tin hôm nay",
    "bóng đá", "football", "soccer", "thể thao", "kết quả bóng đá",
    "nấu ăn", "cooking", "recipe", "công thức nấu",
    "bitcoin", "ethereum", "crypto", "cryptocurrency", "tiền điện tử",
    "viết thơ", "write a poem", "kể chuyện", "tell a story",
    "dịch đoạn văn", "translate this text", "dịch bài văn",
    "bài hát", "song lyrics", "ca từ", "nhạc phim",
    "du lịch đến đâu", "travel recommendation", "nên đi đâu chơi",
    "mua gì", "shopping", "sản phẩm nào tốt",
    "hỏi về lịch sử", "lịch sử thế giới",
]

BOT_INFO_RESPONSE = """Tôi là **Data Analysis AI Agent** 🤖 — trợ lý phân tích dữ liệu thông minh.

## Tôi có thể làm gì?
- 📊 **Phân tích dữ liệu**: Tổng hợp, so sánh, xếp hạng, lọc theo bất kỳ chiều nào
- 📈 **Vẽ biểu đồ tự động**: Bar chart, line chart theo nội dung câu hỏi
- 🔍 **Truy vấn SQL nhanh**: Chạy SELECT an toàn trực tiếp trên dataset
- 🤖 **Agent đa bước**: Phân tích phức tạp, tự gọi nhiều công cụ liên tiếp
- 📝 **Xuất báo cáo**: Markdown report đầy đủ

## Cách sử dụng:
1. **Upload** file CSV hoặc XLSX (tối đa 10MB)
2. **Đặt câu hỏi** bằng tiếng Việt hoặc tiếng Anh
3. **Chọn chế độ**: 📊 Phân tích | ⚡ SQL | 🤖 Agent

## Ví dụ câu hỏi:
- *"Tổng revenue theo category?"*
- *"Top 5 khách hàng theo doanh thu?"*
- *"Trend doanh thu theo tháng?"*
- *"So sánh Q1 và Q2?"*"""

OFF_TOPIC_RESPONSE = (
    "Tôi là trợ lý **phân tích dữ liệu** và chỉ có thể trả lời câu hỏi "
    "liên quan đến file dữ liệu bạn đã upload. "
    "Hãy đặt câu hỏi về dataset của bạn nhé! 📊"
)


# ── Mẫu cấu trúc ────────────────────────────────────────────────────────────
#
# Danh sách chuỗi cố định bên trên khớp quá khít: "viết thơ" không bắt được
# "viết cho tôi một bài thơ", "nấu ăn" không bắt được "cách nấu món phở".
# Ba nhóm regex dưới đây mô tả *dạng câu* thay vì từng câu chữ.

# Câu hỏi mà chủ thể là chính công cụ: <công cụ> ... <khả năng>
_TOOL_SUBJECT = r"(công cụ|cong cu|ứng dụng|ung dung|hệ thống|he thong|phần mềm|phan mem|chương trình|tool|app|bot|trợ lý|tro ly|bạn|ban|em)"
_TOOL_ABILITY = (
    r"(có thể|co the|làm được|lam duoc|hỗ trợ|ho tro|đọc được|doc duoc|xuất|xuat"
    r"|vẽ|ve |dùng|dung |sử dụng|su dung|phát triển|phat trien|tạo ra|tao ra"
    r"|viết ra|chạy được|chay duoc|support|can |does |do you|draw|read|export)"
)

_BOT_INFO_PATTERNS = [
    # "công cụ này có thể …", "app này đọc được …", "bạn hỗ trợ …"
    rf"{_TOOL_SUBJECT}\s*(này|nay|đó)?\s*[^.?!]{{0,28}}?{_TOOL_ABILITY}",
    # "ai là người phát triển/tạo ra ứng dụng này", "who made this app"
    rf"(ai (là người |da |đã )?(phát triển|phat trien|tạo|tao|làm|lam|viết|viet)|who (made|created|built|develops?))\s*[^.?!]{{0,20}}{_TOOL_SUBJECT}",
    # "tôi dùng phần mềm này thế nào", "cách dùng app"
    rf"(dùng|dung|sử dụng|su dung|use)\s*[^.?!]{{0,20}}{_TOOL_SUBJECT}",
]

# Yêu cầu sáng tác, hoặc tư vấn đời sống — không liên quan tới file dữ liệu.
_OFF_TOPIC_PATTERNS = [
    # "viết cho tôi một bài thơ", "sáng tác giúp tôi câu chuyện", "write me a haiku"
    r"(viết|viet|sáng tác|sang tac|kể|ke |làm|lam|soạn|write|compose|tell)"
    r"[^.?!]{0,24}"
    r"(bài thơ|bai tho|thơ|tho |truyện|truyen|câu chuyện|cau chuyen|bài hát|bai hat"
    r"|bài văn|bai van|haiku|poem|story|song|essay)",
    # "nên mua gì", "nên đầu tư vào đâu", "nên đi đâu" — tư vấn tiêu dùng
    r"(nên|nen) (mua|đầu tư|dau tu|đi|di |chọn|chon|học|hoc)\b",
    # "cách làm bánh", "cách nấu phở", "chỉ tôi cách …"
    r"(cách|cach|chỉ tôi|chi toi|dạy tôi|day toi|hướng dẫn tôi cách)\s*(làm|lam|nấu|nau|chế biến|che bien)\b",
    # kiến thức phổ thông: "thủ đô của", "dân số", "ai là tác giả", "who won"
    r"(thủ đô|thu do|dân số|dan so|diện tích|dien tich|ai là tác giả|ai la tac gia"
    r"|ai là người phát minh|thuyết tương đối|thuyet tuong doi"
    r"|who won|capital of|population of)",
    # "tối nay ăn gì", "trưa nay ăn gì"
    r"(ăn gì|an gi)\b",
]

# Tín hiệu cho thấy câu hỏi thật sự nhắm vào dữ liệu đang phân tích.
# Dùng để PHỦ QUYẾT off_topic: "Năm nay có nên mua vàng?" là off-topic, nhưng
# "Nên mua thêm sản phẩm nào dựa trên doanh số?" thì không.
_DATA_SIGNALS = [
    "doanh thu", "doanh so", "doanh số", "lợi nhuận", "loi nhuan", "chi phí", "chi phi",
    "tồn kho", "ton kho", "đơn hàng", "don hang", "khách hàng", "khach hang",
    "cột", "cot ", "dòng", "hàng nào", "bảng", "bang ", "biểu đồ", "bieu do",
    "tổng", "tong ", "trung bình", "trung binh", "số lượng", "so luong",
    "top ", "cao nhất", "cao nhat", "thấp nhất", "thap nhat", "theo tháng",
    "theo quý", "theo vùng", "theo khu vực", "trong file", "trong bảng",
    "dữ liệu", "du lieu", "dataset", "revenue", "sales", "profit", "average",
    "total", "count", "column", "rows",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _has_data_signal(text: str) -> bool:
    return any(sig in text for sig in _DATA_SIGNALS)


def classify_query(question: str) -> str:
    """
    Returns 'bot_info' | 'off_topic' | 'data_summary' | 'data_query'.
    Rule-based, không gọi LLM.

    Thứ tự ưu tiên có chủ đích: câu chào → hỏi về công cụ → ngoài phạm vi →
    tóm tắt dữ liệu → mặc định là câu hỏi dữ liệu. Mặc định nghiêng về
    `data_query` vì bỏ sót một câu hỏi dữ liệu (trả lời chit-chat thay vì
    phân tích) tốn kém hơn nhiều so với lỡ phân tích một câu linh tinh.
    """
    norm = question.lower().strip()
    # Short greeting messages (≤4 words) — catch "chào", "hello", "hi", "hey"
    words = norm.split()
    if len(words) <= 4 and any(tok == words[0] or norm.startswith(tok) for tok in _GREETING_TOKENS):
        return "bot_info"
    if any(p in norm for p in _BOT_INFO):
        return "bot_info"
    if _matches_any(_BOT_INFO_PATTERNS, norm):
        return "bot_info"
    if any(p in norm for p in _OFF_TOPIC):
        return "off_topic"
    # Mẫu off-topic chỉ được áp dụng khi câu không nhắc gì tới dữ liệu.
    if _matches_any(_OFF_TOPIC_PATTERNS, norm) and not _has_data_signal(norm):
        return "off_topic"
    if any(p in norm for p in _DATA_SUMMARY):
        return "data_summary"
    return "data_query"
