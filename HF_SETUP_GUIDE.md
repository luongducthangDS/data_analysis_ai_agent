# HuggingFace LLM Integration Setup Guide

## 🚀 Quick Setup

### 1. Get HuggingFace API Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (choose "Read" access level for inference)
3. Copy the token

### 2. Create `.env` File

Copy `.env.example` to `.env` and fill in your HF token:

```bash
# Copy the template
copy .env.example .env
```

Edit `.env` and replace `your_huggingface_api_key_here` with your actual token:

```env
HF_TOKEN=hf_your_actual_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HF_API_URL=https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Your HuggingFace API token | (required) |
| `HF_MODEL` | Model ID to use | `mistralai/Mistral-7B-Instruct-v0.2` |
| `HF_API_URL` | HuggingFace Inference API URL | (auto-generated from model) |

### Model Configuration

The system uses **Mistral 7B Instruct** by default:
- ✅ Supports Vietnamese language well
- ✅ Good quality outputs
- ✅ Fast inference time
- ✅ Works with HuggingFace free tier

## 🎯 How It Works

### Insight Generation Flow

```
User asks question about dataset
    ↓
System checks if HF_TOKEN is set
    ↓
If YES → Try AI-powered insights with Mistral
    ↓
If NO or LLM fails → Fall back to template-based insights
    ↓
Return insights to user
```

### LLM Inference

When you ask a question about your dataset:

1. **Context Building**: System creates a context with:
   - Dataset statistics (rows, columns)
   - Column metadata (types, missing values)
   - Numeric summaries (min, max, mean, median)
   - Sample data (first 5 rows)

2. **LLM Processing**: Mistral analyzes the context and answers your question

3. **Response**: You get AI-powered insights in Vietnamese

## 📝 Example Usage

### Upload Dataset
```bash
curl -F "file=@sales_data.xlsx" http://localhost:8000/api/upload
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "question": "doanh thu trung bình theo vùng là bao nhiêu?"
  }'
```

**Response with AI insights:**
```json
{
  "answer": "Dựa trên dữ liệu, doanh thu trung bình theo vùng như sau: ...",
  "profile": {...},
  "charts": [...],
  "report_id": "..."
}
```

## 🔐 Security Notes

- **API Token**: Keep your HF_TOKEN secret. Never commit `.env` to version control
- **Read-Only**: HuggingFace inference tokens should have "Read" access only
- **No Code Execution**: The system only sends data to HuggingFace for inference, no arbitrary code is executed

## 🛠️ Troubleshooting

### `ModuleNotFoundError: No module named 'requests'`
```bash
pip install requests
```

### `HF_TOKEN environment variable not set`
Make sure:
1. `.env` file exists in project root
2. `HF_TOKEN` is set correctly
3. Reload the application after changing `.env`

### `HuggingFace API error: Connection timeout`
- Check your internet connection
- HuggingFace might be temporarily unavailable
- System will automatically fall back to template-based insights

### `401 Unauthorized - Invalid authentication`
- Verify your HF_TOKEN is correct
- Token might have expired, generate a new one
- Make sure token has read access

## 📚 Additional Resources

- HuggingFace Documentation: https://huggingface.co/docs
- Mistral Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- Inference API: https://huggingface.co/docs/hub/guides/inference

## ✅ Fallback Strategy

Even without a valid HF_TOKEN, the system continues to work with template-based analysis:

```
No HF_TOKEN
    ↓
Generate insights from dataset profile
    ↓
Use predefined templates
    ↓
Return basic but useful analysis
```

This ensures the application is always functional, with AI-powered insights as an optional enhancement.
