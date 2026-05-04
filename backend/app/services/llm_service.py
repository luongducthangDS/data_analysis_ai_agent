from __future__ import annotations

import os

import requests


class HFLLMClient:
    """HuggingFace Inference API client for optional narrative analysis."""

    def __init__(self):
        self.api_token = os.getenv("HF_TOKEN", "")
        self.model_id = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        self.api_url = os.getenv("HF_API_URL", f"https://api-inference.huggingface.co/models/{self.model_id}")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        if not self.api_token:
            raise ValueError("HF_TOKEN environment variable not set.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.35,
        top_p: float = 0.9,
    ) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "return_full_text": False,
            },
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=45,
            )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return str(first["generated_text"]).strip()

            if isinstance(result, dict) and "generated_text" in result:
                return str(result["generated_text"]).strip()

            return str(result)

        except requests.exceptions.RequestException as exc:
            raise ValueError(f"HuggingFace API error: {exc}") from exc
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ValueError(f"Error parsing HuggingFace response: {exc}") from exc

    def generate_insights(
        self,
        context: str,
        max_tokens: int = 700,
        temperature: float = 0.35,
    ) -> str:
        system_prompt = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Nhiệm vụ: chuyển các số liệu đã tính sẵn thành bản brief điều hành ngắn gọn, có ưu tiên.
Nguyên tắc:
- Không bịa số liệu ngoài context.
- Nêu phát hiện chính, rủi ro dữ liệu và hành động đề xuất.
- Viết tiếng Việt rõ ràng, trực tiếp, phù hợp báo cáo lãnh đạo."""

        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nBản brief:"
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def answer_question(self, question: str, context: str, max_tokens: int = 500) -> str:
        system_prompt = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Trả lời câu hỏi dựa trên context đã tính sẵn. Không bịa số liệu.
Nếu dữ liệu chưa đủ, nói rõ phần còn thiếu và đề xuất bước kiểm tra tiếp theo."""

        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"
        return self.generate(prompt, max_tokens=max_tokens, temperature=0.25)


_client = None


def get_hf_client() -> HFLLMClient:
    global _client
    if _client is None:
        _client = HFLLMClient()
    return _client
