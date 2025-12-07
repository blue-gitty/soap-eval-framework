# 🚀 Quick Start Guide (5 Minutes)

Get the evaluation framework running in 5 minutes.

## Step 1: Clone & Setup (2 min)

```bash
git clone https://github.com/blue-gitty/soap-eval-framework.git
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Configure API Key & Models (1 min)

### 2a. Set API Key

```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key
# Get Gemini key from: https://makersuite.google.com/app/apikey
# Get OpenAI key from: https://platform.openai.com/api-keys
```

Edit `.env`:
```
GEMINI_API_KEY=your-actual-key-here
# OR
OPENAI_API_KEY=your-actual-key-here
```

### 2b. Configure Models (Optional)

Edit `config.yaml` to choose your LLM and embedding models:

```yaml
llm:
  provider: "gemini"  # Options: "ollama", "openai", "gemini"
  model: "gemini-1.5-flash"  # See config.yaml for all options

embeddings:
  provider: "gemini"  # Options: "gemini", "openai", "local"
  model: "text-embedding-004"  # See config.yaml for all options
```

**Default**: Uses Ollama (local) if no API keys are set. Set API keys to use cloud providers.

## Step 3: Validate (30 sec)

```bash
python src/config_validator.py
```

Should show: ✅ All checks passed

## Step 4: Run Evaluation (1 min)

**Fast Mode** (uses pre-existing results):
```bash
python run_full_eval_suite.py --use-processed --charts
```

**Full Pipeline** (generates new results, takes longer, better if want to compare different LLM performances):
```bash
python run_full_eval_suite.py --limit 10 --charts
```

## Step 5: View Results

Check `reports/executive_dashboard/` for charts and `reports/` for CSV tables.

## Troubleshooting

**"API key not found"**
- Make sure `.env` file exists in project root
- Verify key is set: `python src/config_validator.py`

**"Module not found"**
- Reinstall: `pip install -r requirements.txt`

**Need more help?**
- See [SETUP.md](SETUP.md) for detailed instructions
- See [README.md](README.md) for full documentation

