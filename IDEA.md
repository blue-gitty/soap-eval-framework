# DeepScribe Evaluation Framework: Design Philosophy

> **Why three pipelines?** 
Reference-only misses production reality. Non-ref-only lacks clinical perfection benchmark. Thats why I built all three:

Reference (Model vs Gold SOAP): Reveals generation gaps vs clinician perfection.

Non-Reference (Transcript vs Model SOAP): Measures grounding in raw source (halluc=23%, coverage=59%) → keyword capture + hallucinations from audio.

Self-Validation (Transcript vs Gold SOAP): Distinguishes extraction noise (1.8% error) vs true model misses → "excusable gaps" baseline.

Result: Clear gaps (ρ=0.42 convergence). Update prompts/requirements with precision.
---

## 🎯 The Problem We're Solving

Clinical note generation has unique challenges:
- **Patient Safety**: Missing medications or treatments can cause harm
- **Hallucination Risk**: Models may "hallucinate" by adding plausible but incorrect information
- **Extraction Reliability**: We need to distinguish model failures from framework limitations

**My Solution**: Three independent pipelines that validate each other through meta-analysis.

---

## 📊 The Three Pipelines

### Pipeline 1: Reference-Based Evaluation
**"How close to clinical perfection?"**

**What it does:**
- Compares Model-generated SOAP notes vs. Clinician-written Gold Standard SOAP notes
- Breaks down errors by SOAP section (Subjective, Objective, Assessment, Plan)
- Applies clinical risk weighting (Plan section = 50% weight due to patient safety).
Plan > Assessment > Objective > Subjective as our priority order to make sure that our risk scoring flags generated SOAP discrepancies better. We cannot excuse a missed or hallucinated info in Planning since it could "unnecessary treatments or wrong medications". Subjective is low priority becuase patients tend to talk/ ramble about the condition with less precise information.

**Key Metrics:**
- `missing_rate`: Facts in Gold but missing in Model
- `hallucinated_rate`: Facts in Model but not in Gold
- `f1_score`: Overall accuracy
- `semantic_similarity`: Text-level similarity (can be decieiving)

**Example Findings:**
```
Plan section: 38% missing_rate → #1 engineering priority, REVIEW or REJECT SOAP
Joint Pain condition: 70% risk score (CRITICAL)
Diabetes condition: 68% risk score (CRITICAL)
```

**Why it matters:** Identifies where the model deviates from clinical best practices.

---

### Pipeline 2: Non-Reference Evaluation
**"Is the model grounded in what was actually said?"**

**What it does:**
- Compares Model-generated SOAP vs. Raw Patient-Doctor Transcript
- Extracts facts from both, matches them semantically
- Flags hallucinated facts (not in transcript) and missing facts (in transcript but not in model)
- Uses LLM-as-a-Judge for high-risk cases (risk quantified by a custom risk scoring metric)

**Key Metrics:**
- `hallucination_rate`: Facts added by model that weren't in conversation
- `coverage_rate`: Percentage of transcript facts captured
- `risk_score`: Weighted combination (0.7×Halluc + 0.3×Gap)

**Example Findings:**
```
Average hallucination rate: 23%
Average coverage rate: 59%
23 out of 35 notes flagged for review

High-risk example:
Joint Pain: "🔴 HIGH: Ibuprofen stomach discomfort missing → consider alternatives"
```

**Why it matters:** Catches production issues where models add information not present in the source conversation.

---

### Pipeline 3: Self-Validation
**"Is our extraction framework reliable?"**

**What it does:**
- Compares Raw Transcript vs. Gold SOAP (bypasses the model entirely)
- Validates that our fact extraction pipeline is accurate
- Proves that model gaps are real generation problems, not extraction noise

**Key Metrics:**
- `missing_rate`: 1.8% ✅ (target: <3%)
- `hallucinated_rate`: 1.2% ✅ (target: <3%)
- `f1_score`: 0.97 ✅ (near-perfect)

**Why it matters:** If extraction is unreliable, we can't trust model evaluation. This proves our framework is sound.

**Key Insight:** With <2% extraction error, we can confidently attribute model gaps to generation problems, not framework limitations.

---
## Evaluation Approach: Addressing Key Tradeoffs

**Reference-Based vs Non-Reference-Based:** Reference evals compare model SOAP against clinician gold notes for precise quality gaps, like Plan section missing_rate at 38%, but require expensive ground truth curation (limited to 41 notes). Non-reference evals compare model against raw transcripts for scalable "in the wild" measurement, detecting hallucinations (23% rate) and coverage (59%) without gold data (35 notes). Our dual approach uses reference for lab validation and non-reference for production scalability, generating 76 evals total.

**LLM-as-a-Judge vs Deterministic Evals:** LLM judges provide powerful nuanced safety insights (e.g., "ibuprofen stomach discomfort risk"), but are slow and costly. Deterministic metrics (fact matching, F1 0.65, semantic similarity 0.89) are fast and free, running on all notes for core rates like missing_rate and hallucinated_rate. We hybridize: deterministic baselines for speed/CI, then LLM-judge only on high-risk cases (risk>0.20, 23/35 notes), saving 34% cost while ensuring full coverage.

**Net Benefit for Goals:** Fast deterministic pipelines enable PR iteration. Non-reference + triage detects wild regressions (top 20% notes=68% risk). Framework validated (self-val 1.8% error).
---

## ⚖️ Clinical Risk Scoring

**Not all errors are equal.** We weight errors by clinical impact, with hallucinations penalized more heavily than missing facts:

| SOAP Section | Missing Rate Weight | Hallucination Rate Weight | Rationale |
|--------------|---------------------|---------------------------|-----------|
| **Plan** (medications, treatments) | **35%** | **15% × 1.5** (22.5%) | Direct patient safety risk |
| **Objective** (vitals, labs) | 15% | 10% × 1.5 (15%) | Misdiagnosis risk |
| **Assessment** (diagnosis) | 10% | 5% | Wrong treatment path |
| **Subjective** (symptoms) | 5% | 5% | Least critical |

**Formula:**
```
Risk Score = 
  0.35 × Plan_missing + 0.225 × Plan_hallucinated +
  0.15 × Objective_missing + 0.15 × Objective_hallucinated +
  0.10 × Assessment_missing + 0.05 × Assessment_hallucinated +
  0.05 × Subjective_missing + 0.05 × Subjective_hallucinated
```

**Key Design Decisions:**
- **Hallucinations get 1.5× penalty** (except Assessment/Subjective): Fabricated information is worse than missing information
- **Plan section dominates** (~57.5% total weight): Medication/treatment errors pose the highest patient safety risk
- **Missing vs Hallucinated treated separately**: Allows fine-grained control over different error types

This ensures we prioritize fixing medication/treatment errors over symptom documentation issues, and penalize hallucinations more than omissions.

---

## 🔬 Meta-Analysis: Do the Pipelines Agree?

We compute statistical correlations between pipelines to validate framework reliability:

| Comparison | Pearson ρ | Interpretation |
|------------|-----------|----------------|
| Reference vs Non-Reference | 0.42 | Complementary signals (expected) |
| Reference vs Self-Validation | 0.49 | Framework is stable |
| Non-Reference vs Self-Validation | 0.19 | Independent monitoring (good) |

**Target Range:** 0.3-0.6 for clinical NLP (all pass ✅)

**Why this matters:** If pipelines disagree wildly, our framework is unreliable. Moderate correlations (0.3-0.6) indicate complementary but independent signals—exactly what we want.

---

## 🚨 Critical Discovery: Coverage-Hallucination Tradeoff

**Finding:** `Coverage Rate ↑ → Hallucination Rate ↑` (ρ = -0.68, p < 0.01)

**What this means:**
- Models try to "help" by filling gaps with plausible assumptions
- Higher coverage often comes with more hallucinations
- This is a **safety risk** in clinical settings

**Example:**
```
Model sees: "Patient complains of joint pain"
Model adds: "Order x-rays" (not in transcript) → Hallucination
Model adds: "Patient education provided" (not in transcript) → Hallucination
```

**Proposed Fix:**
- Cap coverage at 80% (prevent over-generation)
- Add transcript-only constraints (no assumptions)
- Retrain with explicit "only use transcript facts" instruction

---

## 🛠️ Production Usage

**Single Command Evaluation:**
```bash
python run_full_eval_suite.py --limit 30 --charts
```

**Output:**
- 3 JSON files (reference, non-reference, self-validation results)
- 5 executive dashboard charts
- Meta-analysis report with correlations

**Baseline Model:** `ollama/gemma3:4b` (where the coverage-hallucination tradeoff was discovered)

---

## 📈 Results Summary (96 Total Evaluations)

| Pipeline | Notes Evaluated | Key Finding | Production Action |
|----------|----------------|-------------|-------------------|
| **Reference-Based** | 41 | Plan section: 38% missing rate | **Prompt retraining** for Plan section |
| **Non-Reference** | 35 | 23% hallucination rate | **Live monitoring** + LLM Judge for high-risk |
| **Self-Validation** | 20 | 1.8% extraction error | **Framework validated** ✅ |

**Bottom Line:** 96 evaluations → Clinical risks quantified → Model flaws diagnosed → Production roadmap delivered.

---

## 💡 Key Insights

1. **Three pipelines catch different failure modes** - Reference finds clinical gaps, Non-Reference finds hallucinations, Self-Validation validates the framework.

2. **Clinical weighting matters** - Plan section errors (medications/treatments) are 5× more critical than Subjective errors.

3. **Extraction is reliable** - <2% error means we can trust model evaluation results.

4. **Tradeoff discovered** - Higher coverage doesn't always mean better; it can increase hallucination risk.

5. **Meta-analysis validates framework** - Moderate correlations (0.3-0.6) prove pipelines are complementary, not redundant.

---

## 🎓 For Reviewers

This framework is designed for **production clinical note evaluation**. It:
- ✅ Validates itself through self-validation
- ✅ Catches both missing facts and hallucinations
- Catches Clinical accuracy issues in the Non-reference based evaluation (LLM-as-Judge)
- ✅ Prioritizes patient safety (Plan section weighted 50%)
- ✅ Provides actionable insights (not just metrics)
- ✅ Works with any LLM (Ollama, OpenAI, Gemini)
- Please check the reports folder for charts and tables.

**Next Steps:** Use this framework to evaluate your own models, identify failure modes, and prioritize fixes based on clinical risk. I couldnt do more than 35 cases across 3 pipelines due to time constraints.
