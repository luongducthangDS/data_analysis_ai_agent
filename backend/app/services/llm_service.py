from __future__ import annotations

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Groq client (primary — free, fast, no credit card required)
# ---------------------------------------------------------------------------

class GroqLLMClient:
    """Groq API client using llama-3.3-70b-versatile."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set.")

    def generate(self, prompt: str, max_tokens: int = 700, temperature: float = 0.35, top_p: float = 1.0) -> str:
        from groq import Groq
        client = Groq(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()

    def generate_insights(self, context: str, max_tokens: int = 700, temperature: float = 0.35) -> str:
        system = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Nhiệm vụ: chuyển các số liệu đã tính sẵn thành bản brief điều hành ngắn gọn, có ưu tiên.
Nguyên tắc:
- Không bịa số liệu ngoài context.
- Nêu phát hiện chính, rủi ro dữ liệu và hành động đề xuất.
- Viết tiếng Việt rõ ràng, trực tiếp, phù hợp báo cáo lãnh đạo."""
        return self.generate(f"{system}\n\nContext:\n{context}\n\nBản brief:", max_tokens, temperature)

    def answer_question(self, question: str, context: str, max_tokens: int = 500) -> str:
        system = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Trả lời câu hỏi dựa trên context đã tính sẵn. Không bịa số liệu."""
        return self.generate(f"{system}\n\nContext:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:", max_tokens, 0.25)

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> dict:
        from groq import Groq
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = Groq(api_key=self.api_key).chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return {
                "type": "tool_call",
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
                "raw_message": msg,
            }
        return {"type": "text", "content": (msg.content or "").strip()}


# ---------------------------------------------------------------------------
# Gemini client (fallback)
# ---------------------------------------------------------------------------

class GeminiLLMClient:
    """Google Gemini client — fallback when Groq unavailable."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY not set.")

    def generate(self, prompt: str, max_tokens: int = 700, temperature: float = 0.35, top_p: float = 1.0) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            ),
        )
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()

    def generate_insights(self, context: str, max_tokens: int = 700, temperature: float = 0.35) -> str:
        system = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Chuyển số liệu thành brief điều hành ngắn, có ưu tiên. Viết tiếng Việt."""
        return self.generate(f"{system}\n\nContext:\n{context}\n\nBản brief:", max_tokens, temperature)

    def answer_question(self, question: str, context: str, max_tokens: int = 500) -> str:
        system = "Bạn là trợ lý phân tích dữ liệu. Trả lời dựa trên context. Không bịa số."
        return self.generate(f"{system}\n\nContext:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:", max_tokens, 0.25)


# ---------------------------------------------------------------------------
# HuggingFace client (tertiary)
# ---------------------------------------------------------------------------

class HFLLMClient:
    """Hugging Face Router client — tertiary fallback."""

    def __init__(self):
        self.api_token = os.getenv("HF_TOKEN", "")
        self.model_id = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct:fastest")
        self.base_url = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        if not self.api_token:
            raise ValueError("HF_TOKEN environment variable not set.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.35, top_p: float = 1.0) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return (content or "").strip()

    def generate_insights(self, context: str, max_tokens: int = 700, temperature: float = 0.35) -> str:
        system = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Chuyển số liệu thành brief điều hành ngắn. Viết tiếng Việt."""
        return self.generate(f"{system}\n\nContext:\n{context}\n\nBản brief:", max_tokens, temperature)

    def answer_question(self, question: str, context: str, max_tokens: int = 500) -> str:
        system = "Trả lời câu hỏi dựa trên context. Không bịa số."
        return self.generate(f"{system}\n\nContext:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:", max_tokens, 0.25)


# ---------------------------------------------------------------------------
# Factory — Groq → Gemini → HF (with graceful degradation)
# ---------------------------------------------------------------------------

_client = None


def get_llm_client():
    """
    Return best available LLM client.
    Priority: Groq → Gemini → HuggingFace.
    Set LLM_PROVIDER env var to force a specific provider.
    """
    global _client
    if _client is not None:
        return _client

    forced = os.getenv("LLM_PROVIDER", "").lower()

    if forced == "groq" or (not forced and os.getenv("GROQ_API_KEY")):
        try:
            _client = GroqLLMClient()
            return _client
        except Exception:
            pass

    if forced == "gemini" or (not forced and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))):
        try:
            _client = GeminiLLMClient()
            return _client
        except Exception:
            pass

    if os.getenv("HF_TOKEN"):
        try:
            _client = HFLLMClient()
            return _client
        except Exception:
            pass

    raise RuntimeError(
        "Không có LLM provider nào khả dụng. "
        "Set GROQ_API_KEY, GEMINI_API_KEY, hoặc HF_TOKEN."
    )


# Backward-compatible alias used by existing code
def get_hf_client():
    return get_llm_client()
