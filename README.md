# DeepScribe Evaluation Framework

A comprehensive evaluation framework for clinical SOAP note generation, featuring reference-based, non-reference, and self-validation pipelines with meta-analysis.

## 🚀 Quick Start (5 Minutes)

**For fastest setup, see [QUICK_START.md](QUICK_START.md)**

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd deepscribe-evals
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Validate setup
python src/config_validator.py

# 5. Run evaluation (fast mode with pre-existing results)
python run_full_eval_suite.py --use-processed --charts
```

**Documentation**:
- **[QUICK_START.md](QUICK_START.md)** - 5-minute setup guide ⚡
- **[SETUP.md](SETUP.md)** - Detailed setup instructions
- **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** - Advanced configuration

## 📋 Features

### Three Evaluation Pipelines

1. **Reference-Based Evaluation** (`run_reference_based_eval_pipeline.py`)
   - Compares model-generated SOAP notes against clinician gold standard
   - Metrics: F1, precision, recall, semantic similarity (section-level & overall)
   - Risk scoring and categorization

2. **Non-Reference Evaluation** (`run_non_reference_eval.py`)
   - Compares model SOAP against transcript (ground truth)
   - Metrics: Hallucination rate, coverage rate
   - LLM-as-a-judge for high-risk notes
   - Production readiness triage

3. **Self-Validation** (`run_self_validation.py`)
   - Validates extraction reliability (Transcript vs Gold SOAP)
   - Confirms pipeline consistency
   - Overall metrics only

### Meta-Analysis & Visualization

- **Framework Validation**: Pearson correlations, Cohen's Kappa, ICC across pipelines
- **Executive Dashboard**: 7+ charts (waterfall, production matrix, pareto, radar, violin)
- **Health Problem Metrics**: Grouped analysis by condition
- **Correlation Heatmaps**: Inter-pipeline agreement visualization

## 🎯 Usage

### Master Pipeline (Recommended)

```bash
# Fast mode: Use pre-existing processed results
python run_full_eval_suite.py --use-processed --charts

# Full pipeline: Generate new results
python run_full_eval_suite.py --limit 30 --charts

# Force regeneration
python run_full_eval_suite.py --limit 30 --force --charts
```

### Individual Pipelines

```bash
# Reference-based
python src/run_reference_based_eval_pipeline.py --limit 20

# Non-reference
python src/run_non_reference_eval.py --limit 20

# Self-validation
python src/run_self_validation.py --limit 20
```

## 📊 Output Structure

```
results/
├── reference_based_evals.json      # Model vs Gold metrics
├── non_reference_evals.json        # Model vs Transcript metrics
├── self_validation_evals.json      # Transcript vs Gold validation
└── processed/                      # Pre-computed results (fast mode)

reports/
├── executive_dashboard/            # Visual charts (PNG)
│   ├── risk_waterfall.png
│   ├── production_matrix.png
│   ├── pareto_risk.png
│   └── ...
├── framework_validation_metrics.json  # Meta-analysis
├── executive_validation_table.csv     # Summary table
└── pearson_correlation_matrix.csv     # Correlations
```

## ⚙️ Configuration

### LLM Providers

Edit `config.yaml` to switch providers:

```yaml
llm:
  provider: "gemini"  # Options: "ollama", "openai", "gemini"
  model: "gemini-1.5-flash"
  prompt_strategy: "few-shot"
  temperature: 0.1
```

**Supported Providers:**
- **Ollama** (Local): Free, no API key needed
- **Gemini** (Cloud): Requires `GEMINI_API_KEY`
- **OpenAI** (Cloud): Requires `OPENAI_API_KEY`

### API Keys

Set via environment variables (recommended):

```bash
# .env file
GEMINI_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here  # Optional
```

Or export directly:
```bash
export GEMINI_API_KEY="your-key-here"
```

**Security**: Never commit API keys. The `.env` file is gitignored.

## 📁 Project Structure

```
deepscribe-evals/
├── src/                              # Source code
│   ├── run_reference_based_eval_pipeline.py
│   ├── run_non_reference_eval.py
│   ├── run_self_validation.py
│   ├── viz_utils.py                   # Visualization & meta-analysis
│   ├── section_eval.py                # Section-level evaluation
│   ├── clinical_facts.py              # Fact extraction
│   ├── llm_client.py                 # LLM provider abstraction
│   └── ...
├── results/                           # Evaluation results
│   └── processed/                    # Pre-computed results
├── reports/                          # Generated charts & reports
├── config.yaml                        # Configuration
├── requirements.txt                   # Dependencies
├── .env.example                       # Environment template
├── run_full_eval_suite.py            # Master pipeline
├── SETUP.md                          # Detailed setup guide
└── README.md                         # This file
```

## 🔬 Evaluation Metrics

### Reference-Based Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **Semantic Similarity**: Embedding-based similarity (0-1)
- **Missing Rate**: Facts in gold but not in model
- **Hallucinated Rate**: Facts in model but not in gold
- **Clinical Risk Score**: Weighted risk calculation

### Non-Reference Metrics
- **Hallucination Rate**: Facts in model not in transcript
- **Coverage Rate**: Facts in transcript captured by model
- **Triage Priority**: Risk-based categorization (low/medium/high/critical)
- **Judge Evaluation**: LLM-as-a-judge for high-risk notes

### Meta-Analysis Metrics
- **Pearson Correlation**: Inter-pipeline agreement (target: ρ > 0.75)
- **Cohen's Kappa**: Risk category agreement (target: κ > 0.6)
- **ICC (Intraclass Correlation)**: Reliability measure (target: > 0.8)

## 🛠️ Development

### Running Tests

```bash
# Validate configuration
python src/config_validator.py

# Test individual components
python -c "from src.llm_client import query_llm; print('LLM client OK')"
```

### Adding New Metrics

1. Add metric calculation in `src/section_eval.py`
2. Update result structure in pipeline files
3. Add visualization in `src/viz_utils.py`

## 📚 Documentation

- **[SETUP.md](SETUP.md)**: Complete setup instructions
- **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)**: Advanced configuration & best practices
- **Code Comments**: Inline documentation in all source files

## 🔒 Security

- ✅ API keys stored in `.env` (gitignored)
- ✅ Environment variables take precedence over config.yaml
- ✅ Warnings if keys found in config.yaml
- ✅ No sensitive data in repository

## 📝 License

[Add your license here]

## 📧 Contact

[Add your contact information]

---

**Note for Reviewers**: This repository is set up for easy testing. See [QUICK_START.md](QUICK_START.md) for a 5-minute setup guide. The `--use-processed` flag allows instant results without running the full pipeline.

---

**Built for technical assessment** - Comprehensive evaluation framework for clinical note generation.
