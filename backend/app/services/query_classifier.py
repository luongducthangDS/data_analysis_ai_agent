"""
Query intent classifier.

Classifies user questions into:
  - "bot_info"   → questions about the chatbot itself
  - "off_topic"  → questions unrelated to data analysis
  - "data_query" → questions about the uploaded dataset (default)
"""
from __future__ import annotations

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


def classify_query(question: str) -> str:
    """
    Returns 'bot_info' | 'off_topic' | 'data_query'.
    Rule-based, O(n) keyword scan.
    """
    norm = question.lower().strip()
    # Short greeting messages (≤4 words) — catch "chào", "hello", "hi", "hey"
    words = norm.split()
    if len(words) <= 4 and any(tok == words[0] or norm.startswith(tok) for tok in _GREETING_TOKENS):
        return "bot_info"
    if any(p in norm for p in _BOT_INFO):
        return "bot_info"
    if any(p in norm for p in _OFF_TOPIC):
        return "off_topic"
    return "data_query"
