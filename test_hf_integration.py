#!/usr/bin/env python
"""Test HuggingFace LLM integration"""

import os
import sys
import pandas as pd
from backend.app.services.llm_service import HFLLMClient, get_hf_client
from backend.app.services.profiler import build_profile
from backend.app.services.insights import generate_insights

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Testing HuggingFace LLM Integration\n")

# Check environment
hf_token = os.getenv("HF_TOKEN", "").strip()
print("1. Environment Check")
print(f"   HF_TOKEN: {'SET' if hf_token else 'NOT SET'}")
print(f"   HF_MODEL: {os.getenv('HF_MODEL', 'default')}")

if not hf_token:
    print("\nERROR: HF_TOKEN not set. Please set it in .env file and try again.")
    print("See HF_SETUP_GUIDE.md for setup instructions.")
    exit(1)

# Test LLM Client
print("\n2. Testing LLM Client...")
try:
    client = HFLLMClient()
    print(f"   OK - Client initialized")
    print(f"   - Model: {client.model_id}")
    print(f"   - API URL: {client.api_url}")
except Exception as e:
    print(f"   ERROR - Failed to initialize: {e}")
    exit(1)

# Test Simple Generation
print("\n3. Testing Simple Text Generation...")
try:
    prompt = "Xin chào, bạn là ai?"
    response = client.generate(prompt, max_tokens=100)
    print(f"   OK - Generation successful")
    print(f"   Response: {response[:100]}...")
except Exception as e:
    print(f"   ERROR - Generation failed: {e}")
    exit(1)

# Create test dataset
print("\n4. Creating Test Dataset...")
df = pd.DataFrame({
    "product": ["Laptop", "Điện thoại", "Tablet", "Màn hình", "Bàn phím"],
    "sales": [100, 250, 150, 200, 120],
    "revenue": [1000, 2500, 1500, 2000, 1200],
    "region": ["Hà Nội", "Hà Nội", "TPHCM", "TPHCM", "Đà Nẵng"],
    "quarter": ["Q1", "Q1", "Q1", "Q2", "Q2"]
})

profile = build_profile(df)
print(f"   OK - Dataset created: {len(df)} rows x {len(df.columns)} columns")

# Test Data Context
print("\n5. Testing Data Insights Generation...")
try:
    from backend.app.services.insights import _build_context_for_llm
    
    context = _build_context_for_llm(df, profile, "doanh thu theo vùng là bao nhiêu?")
    print(f"   OK - Context built ({len(context)} chars)")
    print(f"\n   Context preview:")
    for line in context.split("\n")[:8]:
        print(f"   {line}")
except Exception as e:
    print(f"   ERROR - Context building failed: {e}")
    exit(1)

# Test Question Answering
print("\n6. Testing LLM Question Answering...")
try:
    question = "doanh thu trung bình theo vùng là bao nhiêu?"
    answer = client.answer_question(question, context, max_tokens=256)
    print(f"   OK - Answer generated")
    print(f"\n   Question: {question}")
    print(f"   Answer:\n   {answer}")
except Exception as e:
    print(f"   WARNING - Answer generation failed: {e}")
    print(f"   (This might be due to API rate limiting or model availability)")

# Test Full Pipeline
print("\n7. Testing Full Insights Pipeline...")
try:
    insights = generate_insights(df, profile, "bán chạy nhất là sản phẩm nào?")
    print(f"   OK - Full pipeline executed")
    print(f"\n   Insights:")
    print(f"   {insights}")
except Exception as e:
    print(f"   WARNING - Pipeline failed: {e}")

print("\n" + "="*50)
print("HuggingFace integration test completed!")
print("="*50)
print("\nNotes:")
print("- If you see API rate limiting errors, wait a few minutes and retry")
print("- First inference might be slower (model warming up)")
print("- Check HF_SETUP_GUIDE.md for troubleshooting")
