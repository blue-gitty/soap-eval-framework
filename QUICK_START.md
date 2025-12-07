# 🚀 Quick Start Guide (5 Minutes)

Get the evaluation framework running in 5 minutes.

## Step 1: Clone & Setup (2 min)

```bash
git clone <your-repo-url>
cd deepscribe-evals
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Configure API Key (1 min)

```bash
# Copy template
cp .env.example .env

# Edit .env and add your Gemini API key
# Get key from: https://makersuite.google.com/app/apikey
```

Edit `.env`:
```
GEMINI_API_KEY=your-actual-key-here
```

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

**Full Pipeline** (generates new results, takes longer):
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

