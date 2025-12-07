# DeepScribe Evaluation Framework - Setup Guide

Complete setup instructions for reproducing the evaluation environment.

## Prerequisites

- Python 3.8+ (3.9+ recommended)
- pip (Python package manager)
- Git (for cloning the repository)

## Quick Start (5 minutes)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd deepscribe-evals
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

**Option A: Using .env file (Recommended)**

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# Windows: notepad .env
# macOS/Linux: nano .env
```

**Option B: Using Environment Variables**

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your-key-here"
$env:OPENAI_API_KEY="your-key-here"  # Optional

# macOS/Linux
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"  # Optional
```

**Getting API Keys:**
- **Gemini**: https://makersuite.google.com/app/apikey (Required for embeddings)
- **OpenAI**: https://platform.openai.com/api-keys (Optional, only if using OpenAI)

### 5. Configure LLM Provider

Edit `config.yaml` to set your preferred LLM provider:

```yaml
llm:
  provider: "gemini"  # Options: "ollama", "openai", "gemini"
  model: "gemini-1.5-flash"  # Adjust based on provider
```

**Provider Options:**
- **Ollama** (Local, Free): No API key needed. Install Ollama first: https://ollama.ai
- **Gemini** (Cloud): Requires `GEMINI_API_KEY`
- **OpenAI** (Cloud): Requires `OPENAI_API_KEY`

### 6. Validate Setup

```bash
python src/config_validator.py
```

This will check:
- ✅ Python version
- ✅ All dependencies installed
- ✅ API keys configured
- ✅ Configuration file valid

### 7. Run Evaluation (Fast Mode)

**Using pre-existing processed results (instant):**

```bash
python run_full_eval_suite.py --use-processed --charts
```

**Or run full pipeline (takes time, generates new results):**

```bash
python run_full_eval_suite.py --limit 10 --charts
```

## Project Structure

```
deepscribe-evals/
├── src/                          # Source code
│   ├── run_reference_based_eval_pipeline.py    # Reference-based evaluation
│   ├── run_non_reference_eval.py                # Non-reference evaluation
│   ├── run_self_validation.py                   # Self-validation
│   ├── viz_utils.py                             # Visualization & meta-analysis
│   └── ...
├── results/                      # Evaluation results (JSON files)
│   └── processed/               # Pre-computed results for fast mode
├── reports/                      # Generated charts and reports
├── config.yaml                   # Configuration (LLM, embeddings, etc.)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── run_full_eval_suite.py        # Master pipeline script
```

## Usage Examples

### Fast Mode (Use Pre-existing Results)

```bash
# Generate all charts and reports from processed results
python run_full_eval_suite.py --use-processed --charts
```

### Full Pipeline (Generate New Results)

```bash
# Run all 3 pipelines with 30 notes each
python run_full_eval_suite.py --limit 30 --charts

# Run with specific note IDs
python run_full_eval_suite.py --limit 10 --id-list 1 2 3 4 5 --charts

# Force regeneration (ignore processed results)
python run_full_eval_suite.py --limit 30 --force --charts
```

### Individual Pipelines

```bash
# Reference-based evaluation only
python src/run_reference_based_eval_pipeline.py --limit 20

# Non-reference evaluation only
python src/run_non_reference_eval.py --limit 20

# Self-validation only
python src/run_self_validation.py --limit 20
```

## Output Files

After running the evaluation suite, you'll find:

**Results** (`results/`):
- `reference_based_evals.json` - Model vs Gold SOAP metrics
- `non_reference_evals.json` - Model vs Transcript metrics
- `self_validation_evals.json` - Transcript vs Gold validation

**Reports** (`reports/`):
- `executive_dashboard/` - Visual charts (PNG files)
- `framework_validation_metrics.json` - Meta-analysis results
- `executive_validation_table.csv` - Summary table
- `pearson_correlation_matrix.csv` - Correlation matrix

## Troubleshooting

### "API key not found" error

1. Check that `.env` file exists in project root
2. Verify API keys are set correctly
3. Run `python src/config_validator.py` to validate

### "Module not found" error

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "Ollama connection error"

1. Make sure Ollama is installed: https://ollama.ai
2. Start Ollama service: `ollama serve`
3. Pull the model: `ollama pull gemma3:4b` (or your chosen model)

### "Dataset not found" error

The pipeline uses HuggingFace datasets. Ensure you have internet connection on first run.

## Next Steps

- Review `PRODUCTION_GUIDE.md` for advanced configuration
- Check `README.md` for project overview
- Explore `src/viz_utils.py` for visualization customization

## Support

For issues or questions, check:
1. Configuration: `config.yaml` and `src/config_validator.py`
2. Documentation: `README.md` and `PRODUCTION_GUIDE.md`
3. Code comments in source files

