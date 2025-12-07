# Production Best Practices Guide

## 🔐 API Key Management

### Current Status: ✅ Secure
- API keys are loaded from environment variables first
- `.env` file support with automatic loading
- `.env` is gitignored (won't be committed)
- Config file warnings if keys are found in `config.yaml`

### How to Set API Keys

**Option 1: `.env` file (Recommended)**
```bash
# Copy example file
cp .env.example .env

# Edit .env and add your keys:
GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=your-openai-key-here
```

**Option 2: Environment Variables**
```bash
# Linux/Mac
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
$env:OPENAI_API_KEY="your-key-here"

# Windows CMD
set GEMINI_API_KEY=your-key-here
set OPENAI_API_KEY=your-key-here
```

## 🔄 Switching Between Providers

### LLM Providers (for fact extraction and SOAP generation)

Currently supported:
- **Ollama** (local, no API key needed)
- **OpenAI** (cloud, requires API key)
- **Gemini** (cloud, requires API key)

**To switch providers:**

1. Edit `config.yaml`:
```yaml
llm:
  provider: "gemini"  # or "openai" or "ollama"
  model: "gemini-1.5-pro"  # Gemini model, or "gpt-4o-mini" for OpenAI, or "gemma3:4b" for Ollama
```

2. Set API key (if using cloud provider):
```bash
# For Gemini
export GEMINI_API_KEY="your-key"

# For OpenAI
export OPENAI_API_KEY="your-key"
```

3. Verify configuration:
```bash
python src/config_validator.py
```

### Embeddings Providers (for semantic similarity)

Currently supported:
- **Gemini** (requires API key)
- **OpenAI** (requires API key)
- **Local** (SentenceTransformer, no API key)

**To switch embeddings:**

1. Edit `config.yaml`:
```yaml
embeddings:
  provider: "openai"  # or "gemini" or "local"
  model: "text-embedding-3-small"  # OpenAI model
```

2. Set API key (if using cloud provider):
```bash
export GEMINI_API_KEY="your-key"  # for Gemini
# or
export OPENAI_API_KEY="your-key"  # for OpenAI
```

## ✅ Full Provider Support

### LLM Providers
- **Ollama** ✅ (local, no API key)
- **OpenAI** ✅ (cloud, requires API key)
- **Gemini** ✅ (cloud, requires API key) - **Now fully supported!**

### Embeddings Providers
- **Gemini** ✅ (cloud, requires API key)
- **OpenAI** ✅ (cloud, requires API key)
- **Local** ✅ (SentenceTransformer, no API key)

All providers are now fully supported for both LLM calls and embeddings!

## ✅ Configuration Validation

Run health check before starting pipelines:

```bash
python src/config_validator.py
```

This checks:
- ✅ API keys are set (when required)
- ✅ Provider dependencies are installed
- ✅ Configuration values are valid
- ✅ Model names are reasonable

## 🚀 Production Checklist

Before deploying or sharing:

- [ ] API keys moved to `.env` file (not in `config.yaml`)
- [ ] `.env` file is in `.gitignore` ✅ (already done)
- [ ] Run `python src/config_validator.py` - all checks pass
- [ ] Test with your chosen provider/model combination
- [ ] Verify API rate limits are acceptable
- [ ] Check error handling works correctly

## 📝 Provider Comparison

| Provider | Type | API Key | Cost | Speed | Quality | LLM Support | Embeddings Support |
|----------|------|---------|------|-------|---------|-------------|-------------------|
| **Ollama** | Local | ❌ None | Free | Fast | Good | ✅ Yes | ❌ No |
| **OpenAI** | Cloud | ✅ Required | Paid | Fast | Excellent | ✅ Yes | ✅ Yes |
| **Gemini** | Cloud | ✅ Required | Free tier | Fast | Excellent | ✅ Yes | ✅ Yes |

| Embeddings | Type | API Key | Cost | Quality |
|------------|------|---------|------|---------|
| **Local** | Local | ❌ None | Free | Good |
| **Gemini** | Cloud | ✅ Required | Free tier | Excellent |
| **OpenAI** | Cloud | ✅ Required | Paid | Excellent |

## 🔧 Troubleshooting

### "API key not found" error
1. Check `.env` file exists and has correct key
2. Verify environment variable is set: `echo $GEMINI_API_KEY`
3. Restart terminal/IDE after setting env vars
4. Run `python src/config_validator.py` to diagnose

### "Provider not available" error
1. Install missing dependency:
   ```bash
   pip install openai  # for OpenAI
   pip install ollama  # for Ollama
   pip install google-generativeai  # for Gemini
   ```

### Switching providers doesn't work
1. Clear config cache by restarting Python process
2. Verify `config.yaml` has correct provider name
3. Check API key is set for cloud providers
4. Run health check: `python src/config_validator.py`

## 📚 Additional Best Practices

### 1. Rate Limiting
- Consider adding rate limiting for API calls
- Monitor API usage to avoid unexpected costs
- Use local providers (Ollama) for development

### 2. Error Handling
- All API calls have retry logic with exponential backoff
- Errors are logged with context
- Fallback to empty results on failure (configurable)

### 3. Logging
- Consider adding structured logging (e.g., `logging` module)
- Log API calls, errors, and performance metrics
- Use different log levels for dev vs production

### 4. Testing
- Test with different providers before production use
- Validate API keys work with simple test calls
- Monitor for API changes/updates

### 5. Security
- ✅ Never commit API keys to git
- ✅ Use environment variables
- ✅ Rotate keys periodically
- ✅ Use least-privilege API keys when possible

