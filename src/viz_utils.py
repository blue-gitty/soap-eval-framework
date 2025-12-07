
# -----------------------------------------------------------------------------
# DeepScribe Executive Dashboard Logic
# -----------------------------------------------------------------------------

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from math import pi
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

# Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = sns.color_palette("viridis", 8)
RISK_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'] # Green, Yellow, Orange, Red

def get_reports_dir():
    """Get reports directory path consistently - ensures we're in project root, not parent."""
    current_dir = Path(__file__).parent  # src/
    project_root = current_dir.parent     # deepscribe-evals/
    
    # Check if we're in the right place (project root should have results/ and reports/)
    reports_dir = project_root / "reports"
    
    # If reports doesn't exist but we're sure we're in project root, create it
    if not reports_dir.exists() and (project_root / "results").exists():
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir
    
    # Fallback: try to find project root by looking for results/ directory
    if not reports_dir.exists():
        # Look for results directory to identify project root
        if (project_root / "results").exists():
            return reports_dir
        # Try one more level up if needed
        if (project_root.parent / "results").exists():
            return project_root.parent / "reports"
    
    return reports_dir

def get_results_dir():
    """Get results directory path consistently - ensures we're in project root."""
    current_dir = Path(__file__).parent  # src/
    project_root = current_dir.parent     # deepscribe-evals/
    
    results_dir = project_root / "results"
    
    # If results doesn't exist, try to find project root
    if not results_dir.exists():
        # Check if project_root is correct by looking for other markers
        if (project_root / "src").exists() or (project_root / "config.yaml").exists():
            return results_dir
        # Try one more level up if needed
        if (project_root.parent / "results").exists():
            return project_root.parent / "results"
    
    return results_dir

def setup_dirs():
    """Setup output directories."""
    reports_dir = get_reports_dir()
    (reports_dir / "executive_dashboard").mkdir(parents=True, exist_ok=True)

def load_data():
    """Load robust datasets for visualization."""
    results_dir = get_results_dir()
    print(f"📂 Loading data from: {results_dir.resolve()}")
    
    try:
        with open(results_dir / "reference_based_evals.json", 'r') as f:
            ref_data = json.load(f)
            ref_notes = ref_data.get('notes', []) if isinstance(ref_data, dict) else ref_data
            
        with open(results_dir / "non_reference_evals.json", 'r') as f:
            non_ref_data = json.load(f)
            non_ref_notes = non_ref_data.get('notes', []) if isinstance(non_ref_data, dict) else non_ref_data
            
        return pd.DataFrame(ref_notes), pd.DataFrame(non_ref_notes)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# -----------------------------------------------------------------------------
# 1. RISK WATERFALL
# -----------------------------------------------------------------------------
def plot_risk_waterfall(df_ref):
    """Stacked bar chart of Risk contribution by Section."""
    print("Generating Risk Waterfall...")
    
    # Process Data: Calculate contribution of each section to total risk
    # Risk contribution = weight * missing_rate
    rows = []
    for _, row in df_ref.iterrows():
        m = row['metrics']
        rows.append({
            'Condition': row['health_problem'],
            'Plan': 0.35 * m['plan']['missing_rate'] + 0.15 * m['plan']['hallucinated_rate'] * 1.5,
            'Objective': 0.15 * m['objective']['missing_rate'] + 0.10 * m['objective']['hallucinated_rate'] * 1.5,
            'Assessment': 0.10 * m['assessment']['missing_rate'] + 0.05 * m['assessment']['hallucinated_rate'],
            'Subjective': 0.05 * m['subjective']['missing_rate'] + 0.05 * m['subjective']['hallucinated_rate']
        })
    
    df_chart = pd.DataFrame(rows).groupby('Condition').mean()
    
    # Normalize to 100% for stacked "percent" feel or keep raw risk magnitude?
    # Request says "risk %", so let's normalize to show composition 
    # BUT showing raw magnitude is better for risk. Let's do raw stacked (Total height = Avg Risk)
    
    ax = df_chart.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='RdYlBu_r')
    
    plt.title('Executive Risk Waterfall: Where is the Risk Coming From?', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Clinical Risk Contribution', fontsize=12)
    plt.xlabel('Health Condition', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='SOAP Section', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate total risk on top
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='center', color='white', fontsize=9)
        
    plt.tight_layout()
    reports_dir = get_reports_dir()
    (reports_dir / "executive_dashboard").mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "executive_dashboard" / "risk_waterfall.png", dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 2. PRODUCTION MATRIX (Executive Decision View)
# -----------------------------------------------------------------------------
def plot_production_matrix(df_non):
    """Production Decision Matrix: Hallucination (X) vs Coverage (Y) with Clear Zones."""
    print("Generating Production Matrix...")
    
    plt.figure(figsize=(12, 8))
    
    # 1. Define Decision Logic
    def categorize_note(row):
        h = row['hallucination_rate']
        c = row['coverage_rate']
        
        if h < 0.15 and c > 0.80:
            return 'SAFE'
        elif c < 0.60:
            return 'REVIEW'
        elif h > 0.30:
            return 'RETRAIN'
        else:
            return 'REJECT' # Or "MARGINAL"
            
    df_non['decision'] = df_non.apply(categorize_note, axis=1)
    
    # Calculate Stats for Table
    counts = df_non['decision'].value_counts()
    total_n = len(df_non)
    
    # 2. Add Background Shading (Zones)
    # Safe Zone: x < 0.15, y > 0.8
    plt.axvspan(0, 0.15, ymin=0.8, ymax=1, color='green', alpha=0.1, lw=0)
    plt.text(0.02, 0.95, "SAFE ZONE", color='green', fontweight='bold', alpha=0.6)
    
    # Review Zone: y < 0.6
    plt.axhspan(0, 0.60, color='orange', alpha=0.1, lw=0)
    plt.text(0.02, 0.1, "REVIEW ZONE (Low Coverage)", color='#d35400', fontweight='bold', alpha=0.6)
    
    # Retrain/Danger Zone: x > 0.3
    plt.axvspan(0.30, 0.60, color='red', alpha=0.1, lw=0)
    plt.text(0.40, 0.95, "RETRAIN ZONE", color='red', fontweight='bold', alpha=0.6)
    
    # 3. Scatter Plot
    palette = {'SAFE': 'green', 'REVIEW': '#f39c12', 'RETRAIN': 'red', 'REJECT': 'black'}
    
    sns.scatterplot(
        data=df_non,
        x='hallucination_rate',
        y='coverage_rate',
        hue='decision',
        style='decision',
        s=150, # Fixed large size
        alpha=0.8,
        palette=palette
    )
    
    # 4. Add Summary Table (Top Right)
    stats_text = f"Decisions (N={total_n}):\n"
    stats_text += f"SAFE:    {counts.get('SAFE', 0)}\n"
    stats_text += f"REVIEW:  {counts.get('REVIEW', 0)}\n"
    stats_text += f"RETRAIN: {counts.get('RETRAIN', 0)}\n"
    stats_text += f"REJECT:  {counts.get('REJECT', 0)}"
    
    plt.gca().text(
        0.58, 0.75, 
        stats_text, 
        fontsize=11, 
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    )

    plt.title(f'Production Readiness Matrix (N={total_n})', fontsize=16, fontweight='bold')
    plt.xlabel('Hallucination Rate (Lower is Better)', fontsize=12)
    plt.ylabel('Coverage Rate (Higher is Better)', fontsize=12)
    plt.xlim(0, 0.6)
    plt.ylim(0, 1.05)
    
    # Custom Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Decision Category")
    
    plt.tight_layout()
    reports_dir = get_reports_dir()
    (reports_dir / "executive_dashboard").mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "executive_dashboard" / "production_matrix.png", dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 3. PARETO RISK CURVE (Production Triage Efficiency)
# -----------------------------------------------------------------------------
def plot_pareto_curve(df_non):
    """Pareto chart of Triage Risk Concentration (Simulated Production Review)."""
    print("Generating Pareto Curve (Non-Ref)...")
    
    # Needs extracted risk score
    if 'triage' in df_non.columns:
        # Extract scalar if needed, or use column if already flat
        scores = df_non['triage'].apply(lambda x: x.get('risk_score', 0) if isinstance(x, dict) else x)
    else:
        scores = df_non.get('risk_score', pd.Series([0]*len(df_non)))

    # Sort notes by risk desc
    sorted_risk = scores.sort_values(ascending=False).values
    total_risk = sorted_risk.sum()
    
    if total_risk == 0:
        print("Skipping Pareto: Total risk is 0")
        return

    cumulative_risk = sorted_risk.cumsum() / total_risk * 100
    x_pct = np.arange(1, len(sorted_risk) + 1) / len(sorted_risk) * 100
    
    plt.figure(figsize=(10, 6))
    
    # Plot Line
    plt.plot(x_pct, cumulative_risk, marker='o', markersize=4, linewidth=2, color='#2c3e50')
    
    # Reference Lines
    plt.axhline(80, color='red', linestyle='--', alpha=0.7, label='80% Risk Capture')
    
    # Find X where Y=80 (approx)
    idx_80 = np.where(cumulative_risk >= 80)[0]
    if len(idx_80) > 0:
        pct_files_for_80 = x_pct[idx_80[0]]
        plt.axvline(pct_files_for_80, color='red', linestyle='--', alpha=0.7)
        plt.annotate(f"Reviewing top {pct_files_for_80:.1f}% of notes\ncatches 80% of identified risk!",
                     xy=(pct_files_for_80, 80), xytext=(pct_files_for_80+10, 70),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('Pareto Risk Curve: Production Review Efficiency', fontsize=16, fontweight='bold')
    plt.xlabel('% of Notes Reviewed (Sorted by Triage Risk)', fontsize=12)
    plt.ylabel('Cumulative % of Total Risk Captured', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    reports_dir = get_reports_dir()
    (reports_dir / "executive_dashboard").mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "executive_dashboard" / "pareto_risk.png", dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 3b. VIOLIN PLOT (Domain Monitoring)
# -----------------------------------------------------------------------------
def plot_violin_risk(df_non):
    """Violin plot of Hallucination Rates by Condition."""
    print("Generating Violin Plot...")
    
    # Validate input
    if df_non.empty:
        print("  ⚠️  Skipping violin plot: No data available")
        return
    
    if 'health_problem' not in df_non.columns or 'hallucination_rate' not in df_non.columns:
        print("  ⚠️  Skipping violin plot: Required columns missing")
        return
    
    # Filter out NaN values
    df_clean = df_non.dropna(subset=['health_problem', 'hallucination_rate'])
    if df_clean.empty:
        print("  ⚠️  Skipping violin plot: No valid data after cleaning")
        return
    
    # Check if we have multiple conditions
    unique_conditions = df_clean['health_problem'].nunique()
    if unique_conditions < 2:
        print(f"  ⚠️  Skipping violin plot: Need at least 2 conditions, found {unique_conditions}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Sort by median hallucination rate
    try:
        order = df_clean.groupby('health_problem')['hallucination_rate'].median().sort_values(ascending=False).index
    except Exception as e:
        print(f"  ⚠️  Error calculating order: {e}")
        order = df_clean['health_problem'].unique()
    
    sns.violinplot(data=df_clean, x='health_problem', y='hallucination_rate', order=order, color='lightblue', inner='stick')
    
    plt.title('Hallucination Rate Distribution by Condition (Violin)', fontsize=16, fontweight='bold')
    plt.ylabel('Hallucination Rate')
    plt.xlabel('Condition')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    reports_dir = get_reports_dir()
    (reports_dir / "executive_dashboard").mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "executive_dashboard" / "violin_condition.png", dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 4. CONDITION RADAR CHARTS (Hallucination & Missingness)
# -----------------------------------------------------------------------------
def plot_radar_charts(df_ref, metric='hallucinated_rate', title_suffix='Hallucination'):
    """Radar charts for top conditions showing specific metric per section."""
    print(f"Generating Radar Charts ({title_suffix})...")
    
    # Prep Data
    rows = []
    for _, row in df_ref.iterrows():
        m = row['metrics']
        values = [
            m['subjective'].get(metric, 0),
            m['objective'].get(metric, 0),
            m['assessment'].get(metric, 0),
            m['plan'].get(metric, 0)
        ]
        rows.append({'Condition': row['health_problem'], 'Values': values})
    
    # Aggregate by condition - Take Top 6 most frequent
    top_conditions = df_ref['health_problem'].value_counts().head(6).index.tolist()
    
    if not top_conditions: return

    # Setup Plot
    categories = ['Subjective', 'Objective', 'Assessment', 'Plan']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Custom vibrant palette
    colors = sns.color_palette("bright", len(top_conditions))
    
    for i, cond in enumerate(top_conditions):
        # Get mean values for this condition
        cond_rows = [r['Values'] for r in rows if r['Condition'] == cond]
        if not cond_rows: continue
        
        avg_values = np.mean(cond_rows, axis=0).tolist()
        avg_values += avg_values[:1] # Close the loop
        
        ax.plot(angles, avg_values, linewidth=2, linestyle='solid', label=cond, color=colors[i])
        ax.fill(angles, avg_values, color=colors[i], alpha=0.1)

    plt.xticks(angles[:-1], categories, fontsize=12, fontweight='bold')
    
    # Y-labels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["10%","20%","30%","40%","50%"], color="grey", size=8)
    plt.ylim(0, 0.6) # Fixed scale for comparison
    
    plt.title(f'Profile by Condition: {title_suffix} Rate', fontsize=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Top Conditions")
    
    reports_dir = get_reports_dir()
    (reports_dir / "executive_dashboard").mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "executive_dashboard" / f"radar_{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_radars(df_ref):
    """Generate both Hallucination and Missingness Radars."""
    plot_radar_charts(df_ref, metric='hallucinated_rate', title_suffix='Safety Risk (Hallucination)')
    plot_radar_charts(df_ref, metric='missing_rate', title_suffix='Quality Gap (Missingness)')

# -----------------------------------------------------------------------------
# 5. EVAL VALIDATION BAR
# -----------------------------------------------------------------------------
def plot_eval_validation(df_ref, df_non):
    """Comparison of Reference Metrics vs Non-Reference Metrics."""
    print("Generating Validation Bar...")
    
    # Calculate Averages
    ref_halluc = df_ref['metrics'].apply(lambda x: x['overall']['hallucinated_rate']).mean()
    non_ref_halluc = df_non['hallucination_rate'].mean()
    
    # Ideally we compare GroundTruth (Transcript) too if we had it separately stats
    # For now compare Ref vs Non-Ref pipeline
    
    data = {
        'Pipeline': ['Reference-Based (Gold)', 'Non-Reference (Proxy)'],
        'Avg Hallucination Rate': [ref_halluc, non_ref_halluc],
        'Avg Coverage/Recall': [
             df_ref['metrics'].apply(lambda x: x['overall']['recall']).mean(),
             df_non['coverage_rate'].mean()
        ]
    }
    
    df_plot = pd.DataFrame(data).melt(id_vars='Pipeline', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x='Metric', y='Score', hue='Pipeline', palette='muted')
    
    # Add green zone
    plt.axhspan(0, 0.1, color='green', alpha=0.1, label='Target Zone (<10%)')
    
    plt.title('Pipeline Validation: Proxy vs Gold Standard', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.legend()
    
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f')
        
    plt.tight_layout()
    reports_dir = get_reports_dir()
    (reports_dir / "executive_dashboard").mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "executive_dashboard" / "eval_validation.png", dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 6. HEALTH PROBLEM GROUPED METRICS TABLE
# -----------------------------------------------------------------------------
def generate_health_problem_metrics_table(df_ref, metric='semantic_similarity'):
    """
    Generate grouped metrics table by health problem and section.
    
    Args:
        df_ref: DataFrame with reference-based eval results
        metric: Metric to aggregate ('semantic_similarity', 'f1', 'missing_rate', etc.)
    
    Returns:
        DataFrame with columns: Condition, Section, Avg Similarity, Max Similarity
    """
    rows = []
    
    for _, row in df_ref.iterrows():
        health_problem = row['health_problem']
        metrics = row['metrics']
        
        sections = ['subjective', 'objective', 'assessment', 'plan']
        for section in sections:
            section_metric = metrics.get(section, {})
            value = section_metric.get(metric, 0.0)
            
            rows.append({
                'Condition': health_problem,
                'Section': section.capitalize(),
                metric: value
            })
    
    df_long = pd.DataFrame(rows)
    
    if df_long.empty:
        return pd.DataFrame()
    
    # Group by Condition and Section, calculate avg and max
    grouped = df_long.groupby(['Condition', 'Section'])[metric].agg(['mean', 'max']).reset_index()
    
    # Rename columns for display (simplified names)
    if metric == 'semantic_similarity':
        avg_col = 'Avg Similarity'
        max_col = 'Max Similarity'
    else:
        avg_col = f'Avg {metric.replace("_", " ").title()}'
        max_col = f'Max {metric.replace("_", " ").title()}'
    
    grouped.columns = ['Condition', 'Section', avg_col, max_col]
    
    # Round to 4 decimal places
    grouped[avg_col] = grouped[avg_col].round(4)
    grouped[max_col] = grouped[max_col].round(4)
    
    # Sort by Condition, then by Section order
    section_order = {'Subjective': 0, 'Objective': 1, 'Assessment': 2, 'Plan': 3}
    grouped['Section_Order'] = grouped['Section'].map(section_order)
    grouped = grouped.sort_values(['Condition', 'Section_Order']).drop('Section_Order', axis=1)
    
    return grouped

def print_health_problem_metrics_table(df_ref, metric='semantic_similarity'):
    """
    Print formatted health problem grouped metrics table.
    
    Args:
        df_ref: DataFrame with reference-based eval results
        metric: Metric to aggregate (default: 'semantic_similarity')
    """
    print("\n" + "="*80)
    print(f"📊 HEALTH PROBLEM GROUPED METRICS: {metric.replace('_', ' ').title()}")
    print("="*80)
    
    df_table = generate_health_problem_metrics_table(df_ref, metric)
    
    if df_table.empty:
        print("No data available.")
        return
    
    # Print table with proper formatting
    print(df_table.to_string(index=False))
    print("="*80)
    
    return df_table

def save_health_problem_metrics_table(df_ref, output_path=None, metric='semantic_similarity'):
    """
    Save health problem grouped metrics table to CSV.
    
    Args:
        df_ref: DataFrame with reference-based eval results
        output_path: Path to save CSV (default: reports/health_problem_metrics.csv)
        metric: Metric to aggregate (default: 'semantic_similarity')
    """
    if output_path is None:
        reports_dir = get_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / f"health_problem_{metric}_table.csv"
    
    df_table = generate_health_problem_metrics_table(df_ref, metric)
    
    if not df_table.empty:
        df_table.to_csv(output_path, index=False)
        print(f"✅ Table saved to {output_path}")
        return output_path
    else:
        print("⚠️  No data to save.")
        return None

# -----------------------------------------------------------------------------
# 7. FRAMEWORK VALIDATION METRICS (Inter-Rater Reliability)
# -----------------------------------------------------------------------------
def load_all_eval_data():
    """Load all three evaluation JSON files and normalize field names."""
    results_dir = get_results_dir()
    print(f"📂 Loading evaluation data from: {results_dir.resolve()}")
    
    try:
        # Reference-based
        with open(results_dir / "reference_based_evals.json", 'r') as f:
            ref_data = json.load(f)
            ref_notes = ref_data.get('notes', []) if isinstance(ref_data, dict) else ref_data
        
        # Non-reference
        with open(results_dir / "non_reference_evals.json", 'r') as f:
            non_ref_data = json.load(f)
            non_ref_notes = non_ref_data if isinstance(non_ref_data, list) else non_ref_data.get('notes', [])
        
        # Self-validation
        with open(results_dir / "self_validation_evals.json", 'r') as f:
            self_val_data = json.load(f)
            self_val_notes = self_val_data.get('notes', []) if isinstance(self_val_data, dict) else self_val_data
        
        # Create DataFrames
        df_ref = pd.DataFrame(ref_notes)
        df_non = pd.DataFrame(non_ref_notes)
        df_self = pd.DataFrame(self_val_notes)
        
        # Normalize field names across all DataFrames for correlation
        # Target: hallucination_rate, missing_rate, f1_score (all top-level)
        
        if not df_ref.empty:
            # Ref: Extract from metrics.overall.* → top-level columns
            df_ref['hallucination_rate'] = df_ref['metrics'].apply(
                lambda x: x.get('overall', {}).get('hallucinated_rate', np.nan) if isinstance(x, dict) else np.nan
            )
            df_ref['missing_rate'] = df_ref['metrics'].apply(
                lambda x: x.get('overall', {}).get('missing_rate', np.nan) if isinstance(x, dict) else np.nan
            )
            df_ref['f1_score'] = df_ref['metrics'].apply(
                lambda x: x.get('overall', {}).get('f1', np.nan) if isinstance(x, dict) else np.nan
            )
            print(f"  ✅ Ref: Normalized hallucination_rate, missing_rate, f1_score")
        
        if not df_non.empty:
            # Non-Ref: Already has hallucination_rate (top-level)
            # missing_rate = 1 - coverage_rate
            df_non['missing_rate'] = df_non.apply(
                lambda x: 1 - x.get('coverage_rate', 0) if not pd.isna(x.get('coverage_rate', np.nan)) else np.nan,
                axis=1
            )
            # f1_score not available in non-ref
            df_non['f1_score'] = np.nan
            print(f"  ✅ Non-Ref: Normalized missing_rate (from coverage_rate), f1_score=NaN")
        
        if not df_self.empty:
            # Self-Val: Rename hallucinated_rate → hallucination_rate, f1 → f1_score
            if 'hallucinated_rate' in df_self.columns:
                df_self['hallucination_rate'] = df_self['hallucinated_rate']
            if 'f1' in df_self.columns:
                df_self['f1_score'] = df_self['f1']
            # missing_rate already exists at top-level
            print(f"  ✅ Self-Val: Normalized hallucination_rate (from hallucinated_rate), f1_score (from f1)")
        
        return df_ref, df_non, df_self
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def bin_risk_score(score, thresholds=[0.2, 0.4, 0.6]):
    """Bin risk score into categories: LOW, MODERATE, HIGH, CRITICAL."""
    if pd.isna(score):
        return None
    if score < thresholds[0]:
        return "LOW"
    elif score < thresholds[1]:
        return "MODERATE"
    elif score < thresholds[2]:
        return "HIGH"
    else:
        return "CRITICAL"

def bin_priority(priority):
    """Normalize priority to standard categories."""
    if pd.isna(priority) or priority is None:
        return None
    priority_str = str(priority).lower()
    if 'low' in priority_str:
        return "LOW"
    elif 'medium' in priority_str or 'moderate' in priority_str:
        return "MODERATE"
    elif 'high' in priority_str:
        return "HIGH"
    elif 'critical' in priority_str:
        return "CRITICAL"
    else:
        return "MODERATE"  # Default

def compute_icc(x, y):
    """
    Compute Intraclass Correlation Coefficient (ICC) for two measurements.
    Simplified ICC(2,1) - two-way random effects, single rater.
    """
    # Remove NaN pairs
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = np.array(x[mask])
    y_clean = np.array(y[mask])
    
    if len(x_clean) < 2:
        return None, None
    
    # Mean of both measurements
    mean_all = np.mean(np.concatenate([x_clean, y_clean]))
    
    # Between-subject variance
    means = (x_clean + y_clean) / 2
    ss_between = np.sum((means - mean_all) ** 2) * 2
    
    # Within-subject variance
    ss_within = np.sum((x_clean - means) ** 2) + np.sum((y_clean - means) ** 2)
    
    # Total variance
    ss_total = ss_between + ss_within
    
    if ss_total == 0:
        return None, None
    
    # ICC = (MS_between - MS_within) / (MS_between + MS_within)
    n = len(x_clean)
    ms_between = ss_between / (n - 1) if n > 1 else 0
    ms_within = ss_within / n
    
    if ms_between + ms_within == 0:
        return None, None
    
    icc = (ms_between - ms_within) / (ms_between + ms_within)
    
    # Clamp to [-1, 1]
    icc = max(-1, min(1, icc))
    
    return icc, n

def match_records_by_id_or_health(df1, df2, id_col='id', health_col='health_problem'):
    """Match records between two DataFrames by id (primary) or health_problem (fallback)."""
    merged = []
    
    for idx1, row1 in df1.iterrows():
        id1 = row1.get(id_col)
        health1 = row1.get(health_col)
        
        # Try ID match first
        matches = df2[df2[id_col] == id1] if id1 is not None and not pd.isna(id1) else pd.DataFrame()
        
        # Fallback to health_problem match
        if matches.empty and health1 is not None and not pd.isna(health1):
            matches = df2[df2[health_col] == health1]
        
        if not matches.empty:
            # Take first match
            row2 = matches.iloc[0]
            merged.append({
                'id': id1,
                'health_problem': health1,
                'row1': row1,
                'row2': row2
            })
    
    return merged

def map_risk_category_to_numeric(category):
    """Map risk_category to numeric: {'LOW':0, 'MEDIUM':1, 'HIGH':2, 'CRITICAL':3}"""
    if pd.isna(category) or category is None:
        return None
    category_str = str(category).upper()
    mapping = {'LOW': 0, 'MODERATE': 1, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
    return mapping.get(category_str, None)

def map_priority_to_numeric(priority):
    """Map triage.priority to numeric: {'medium':1, 'high':2, 'critical':3}"""
    if pd.isna(priority) or priority is None:
        return None
    priority_str = str(priority).lower()
    mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
    return mapping.get(priority_str, None)

def plot_pearson_heatmap(pearson_matrix, target_metrics):
    """Generate heatmap of Pearson correlations."""
    # Build matrix: rows = pairings, columns = metrics
    pairings = ['ref_vs_non_ref', 'ref_vs_self_val', 'non_ref_vs_self_val']
    
    matrix_data = []
    row_labels = []
    
    for pairing in pairings:
        row = []
        for metric in target_metrics:
            key = f'{pairing}_{metric}'
            if key in pearson_matrix:
                row.append(pearson_matrix[key]['r'])
            else:
                row.append(np.nan)
        if not all(pd.isna(v) for v in row):
            matrix_data.append(row)
            row_labels.append(pairing.replace('_', ' ').title())
    
    if not matrix_data:
        print("  ⚠️  No data for heatmap")
        return
    
    # Create DataFrame
    df_heatmap = pd.DataFrame(matrix_data, index=row_labels, columns=[m.replace('_', ' ').title() for m in target_metrics])
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', center=0.75, 
                vmin=0, vmax=1, cbar_kws={'label': 'Pearson r'}, linewidths=0.5)
    plt.title('Pearson Correlation Matrix: Framework Validation', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Pairing', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.tight_layout()
    
    # Save chart
    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    chart_path = reports_dir / "meta_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Heatmap saved to {chart_path}")
    plt.close()
    
    # Save correlation matrix as CSV
    csv_path = reports_dir / "pearson_correlation_matrix.csv"
    df_heatmap.to_csv(csv_path, index=True)
    print(f"  ✅ Correlation matrix saved to {csv_path}")

def compute_validation_metrics():
    """
    Compute framework validation metrics: Pearson correlations, Kappa, ICC.
    Returns comprehensive validation report with executive table and heatmap.
    """
    print("\n" + "="*80)
    print("🔬 META-ANALYSIS: Framework Validation (Ref 41 + Non-Ref 35 + Self-Val 20)")
    print("="*80)
    
    # Load all data
    df_ref, df_non, df_self = load_all_eval_data()
    
    if df_ref.empty or df_non.empty or df_self.empty:
        print("⚠️  Missing evaluation data. Cannot compute validation metrics.")
        return None
    
    print(f"\n📊 Dataset Sizes:")
    print(f"   Reference-Based: {len(df_ref)} records")
    print(f"   Non-Reference: {len(df_non)} records")
    print(f"   Self-Validation: {len(df_self)} records")
    
    results = {
        'ref_vs_non_ref': {},
        'ref_vs_self_val': {},
        'non_ref_vs_self_val': {},
        'executive_table': [],
        'pearson_matrix': {},
        'summary': {}
    }
    
    # Target metrics for correlation (using normalized field names)
    target_metrics = ['f1_score', 'hallucination_rate', 'missing_rate', 'semantic_similarity']
    
    # Match all pairs
    matched_ref_non = match_records_by_id_or_health(df_ref, df_non)
    matched_ref_self = match_records_by_id_or_health(df_ref, df_self)
    matched_non_self = match_records_by_id_or_health(df_non, df_self)
    
    # Extract metrics helper functions - using normalized top-level columns
    def extract_ref_metrics(row):
        """Extract from Ref DataFrame - all fields now at top-level after normalization."""
        return {
            'f1_score': row.get('f1_score', np.nan),
            'hallucination_rate': row.get('hallucination_rate', np.nan),
            'missing_rate': row.get('missing_rate', np.nan),
            'semantic_similarity': row.get('metrics', {}).get('overall', {}).get('semantic_similarity', np.nan) if isinstance(row.get('metrics'), dict) else np.nan
        }
    
    def extract_non_metrics(row):
        """Extract from Non-Ref DataFrame - all fields now at top-level after normalization."""
        return {
            'f1_score': row.get('f1_score', np.nan),  # Always NaN for non-ref
            'hallucination_rate': row.get('hallucination_rate', np.nan),
            'missing_rate': row.get('missing_rate', np.nan),
            'semantic_similarity': np.nan  # Non-ref doesn't have semantic_similarity
        }
    
    def extract_self_metrics(row):
        """Extract from Self-Val DataFrame - all fields now at top-level after normalization."""
        return {
            'f1_score': row.get('f1_score', np.nan),
            'hallucination_rate': row.get('hallucination_rate', np.nan),
            'missing_rate': row.get('missing_rate', np.nan),
            'semantic_similarity': row.get('semantic_similarity', np.nan)
        }
    
    # ========================================================================
    # 1. PEARSON CORRELATION MATRIX
    # ========================================================================
    print("\n📊 Computing Pearson Correlation Matrix")
    print("-" * 80)
    
    pearson_matrix = {}
    executive_rows = []
    
    # 1.1 Ref vs Non-Ref
    if matched_ref_non:
        print("\n  Pairing: Ref vs Non-Ref")
        ref_metrics_dict = {m: [] for m in target_metrics}
        non_metrics_dict = {m: [] for m in target_metrics}
        
        for match in matched_ref_non:
            ref_m = extract_ref_metrics(match['row1'])
            non_m = extract_non_metrics(match['row2'])
            for metric in target_metrics:
                ref_metrics_dict[metric].append(ref_m[metric])
                non_metrics_dict[metric].append(non_m[metric])
        
        # Compute Pearson for each metric
        best_pearson = None
        best_metric = None
        for metric in target_metrics:
            ref_arr = np.array(ref_metrics_dict[metric])
            non_arr = np.array(non_metrics_dict[metric])
            mask = ~(pd.isna(ref_arr) | pd.isna(non_arr))
            
            if mask.sum() >= 3:
                r, p = pearsonr(ref_arr[mask], non_arr[mask])
                pearson_matrix[f'ref_vs_non_ref_{metric}'] = {'r': float(r), 'p': float(p), 'n': int(mask.sum())}
                
                if best_pearson is None or abs(r) > abs(best_pearson):
                    best_pearson = r
                    best_metric = metric
                
                print(f"    {metric}: r = {r:.4f} (p={p:.4f}, n={mask.sum()})")
        
        # Risk Category Kappa (Ref risk_category vs Non-Ref triage.priority)
        ref_risk_cats = []
        non_priorities = []
        
        for match in matched_ref_non:
            ref_row = match['row1']
            non_row = match['row2']
            
            # Ref: risk_category
            ref_risk_cat = ref_row.get('risk_category', None)
            ref_num = map_risk_category_to_numeric(ref_risk_cat)
            
            # Non-Ref: triage.priority
            triage = non_row.get('triage', {})
            if isinstance(triage, dict):
                non_priority = triage.get('priority', None)
                non_num = map_priority_to_numeric(non_priority)
            else:
                non_num = None
            
            if ref_num is not None and non_num is not None:
                ref_risk_cats.append(ref_num)
                non_priorities.append(non_num)
        
        kappa_ref_non = None
        if len(ref_risk_cats) >= 2:
            kappa_ref_non = cohen_kappa_score(ref_risk_cats, non_priorities)
            results['ref_vs_non_ref']['risk_category_kappa'] = {
                'kappa': float(kappa_ref_non),
                'n': len(ref_risk_cats),
                'target': 0.6,
                'passed': bool(kappa_ref_non > 0.6)
            }
            print(f"    Risk Category Kappa: κ = {kappa_ref_non:.4f} (n={len(ref_risk_cats)})")
        
        # Add to executive table
        executive_rows.append({
            'Comparison': 'Ref vs Non-Ref',
            'Pearson ρ': f"{best_pearson:.2f}" if best_pearson is not None else "-",
            'Kappa': f"{kappa_ref_non:.2f}" if kappa_ref_non is not None else "-",
            'ICC': "-",
            'Status': '✅' if (best_pearson is not None and abs(best_pearson) > 0.75) and (kappa_ref_non is not None and kappa_ref_non > 0.6) else '❌'
        })
    
    # 1.2 Ref vs Self-Val
    if matched_ref_self:
        print("\n  Pairing: Ref vs Self-Val")
        ref_metrics_dict = {m: [] for m in target_metrics}
        self_metrics_dict = {m: [] for m in target_metrics}
        
        for match in matched_ref_self:
            ref_m = extract_ref_metrics(match['row1'])
            self_m = extract_self_metrics(match['row2'])
            for metric in target_metrics:
                ref_metrics_dict[metric].append(ref_m[metric])
                self_metrics_dict[metric].append(self_m[metric])
        
        # Compute Pearson for each metric
        best_pearson = None
        best_metric = None
        for metric in target_metrics:
            ref_arr = np.array(ref_metrics_dict[metric])
            self_arr = np.array(self_metrics_dict[metric])
            mask = ~(pd.isna(ref_arr) | pd.isna(self_arr))
            
            if mask.sum() >= 3:
                r, p = pearsonr(ref_arr[mask], self_arr[mask])
                pearson_matrix[f'ref_vs_self_val_{metric}'] = {'r': float(r), 'p': float(p), 'n': int(mask.sum())}
                
                if best_pearson is None or abs(r) > abs(best_pearson):
                    best_pearson = r
                    best_metric = metric
                
                print(f"    {metric}: r = {r:.4f} (p={p:.4f}, n={mask.sum()})")
        
        # ICC for overall f1_score
        ref_f1 = []
        self_f1 = []
        for match in matched_ref_self:
            ref_m = extract_ref_metrics(match['row1'])
            self_m = extract_self_metrics(match['row2'])
            ref_f1.append(ref_m['f1_score'])
            self_f1.append(self_m['f1_score'])
        
        ref_f1_arr = np.array(ref_f1)
        self_f1_arr = np.array(self_f1)
        icc, n_icc = compute_icc(ref_f1_arr, self_f1_arr)
        
        icc_value = None
        if icc is not None:
            icc_value = icc
            results['ref_vs_self_val']['f1_icc'] = {
                'icc': float(icc),
                'n': n_icc,
                'target': 0.8,
                'passed': bool(icc > 0.8)
            }
            print(f"    F1 ICC: ICC = {icc:.4f} (n={n_icc})")
        
        # Add to executive table
        executive_rows.append({
            'Comparison': 'Ref vs Self-Val',
            'Pearson ρ': f"{best_pearson:.2f}" if best_pearson is not None else "-",
            'Kappa': "-",
            'ICC': f"{icc_value:.2f}" if icc_value is not None else "-",
            'Status': '✅' if (best_pearson is not None and abs(best_pearson) > 0.75) and (icc_value is not None and icc_value > 0.8) else '❌'
        })
    
    # 1.3 Non-Ref vs Self-Val
    if matched_non_self:
        print("\n  Pairing: Non-Ref vs Self-Val")
        non_metrics_dict = {m: [] for m in target_metrics}
        self_metrics_dict = {m: [] for m in target_metrics}
        
        for match in matched_non_self:
            non_m = extract_non_metrics(match['row1'])
            self_m = extract_self_metrics(match['row2'])
            for metric in target_metrics:
                non_metrics_dict[metric].append(non_m[metric])
                self_metrics_dict[metric].append(self_m[metric])
        
        # Compute Pearson for each metric
        best_pearson = None
        for metric in target_metrics:
            non_arr = np.array(non_metrics_dict[metric])
            self_arr = np.array(self_metrics_dict[metric])
            mask = ~(pd.isna(non_arr) | pd.isna(self_arr))
            
            if mask.sum() >= 3:
                r, p = pearsonr(non_arr[mask], self_arr[mask])
                pearson_matrix[f'non_ref_vs_self_val_{metric}'] = {'r': float(r), 'p': float(p), 'n': int(mask.sum())}
                
                if best_pearson is None or abs(r) > abs(best_pearson):
                    best_pearson = r
                
                print(f"    {metric}: r = {r:.4f} (p={p:.4f}, n={mask.sum()})")
        
        # Risk Category Kappa (Non-Ref triage.priority vs Self-Val proxy)
        non_priorities = []
        self_risk_cats = []
        
        for match in matched_non_self:
            non_row = match['row1']
            self_row = match['row2']
            
            # Non-Ref: triage.priority
            triage = non_row.get('triage', {})
            if isinstance(triage, dict):
                non_priority = triage.get('priority', None)
                non_num = map_priority_to_numeric(non_priority)
            else:
                non_num = None
            
            # Self-Val: proxy from missing_rate + hallucinated_rate
            self_missing = self_row.get('missing_rate', 0)
            self_halluc = self_row.get('hallucinated_rate', 0)
            self_risk_score = (self_missing + self_halluc) / 2
            self_risk_cat = bin_risk_score(self_risk_score)
            self_num = map_risk_category_to_numeric(self_risk_cat)
            
            if non_num is not None and self_num is not None:
                non_priorities.append(non_num)
                self_risk_cats.append(self_num)
        
        kappa_non_self = None
        if len(non_priorities) >= 2:
            kappa_non_self = cohen_kappa_score(non_priorities, self_risk_cats)
            results['non_ref_vs_self_val']['risk_category_kappa'] = {
                'kappa': float(kappa_non_self),
                'n': len(non_priorities),
                'target': 0.6,
                'passed': bool(kappa_non_self > 0.6)
            }
            print(f"    Risk Category Kappa: κ = {kappa_non_self:.4f} (n={len(non_priorities)})")
        
        # Add to executive table
        executive_rows.append({
            'Comparison': 'Non-Ref vs Self-Val',
            'Pearson ρ': f"{best_pearson:.2f}" if best_pearson is not None else "-",
            'Kappa': f"{kappa_non_self:.2f}" if kappa_non_self is not None else "-",
            'ICC': "-",
            'Status': '✅' if (best_pearson is not None and abs(best_pearson) > 0.75) and (kappa_non_self is not None and kappa_non_self > 0.6) else '❌'
        })
    
    results['pearson_matrix'] = pearson_matrix
    results['executive_table'] = executive_rows
    
    # ========================================================================
    # 2. EXECUTIVE TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("📋 EXECUTIVE TABLE")
    print("="*80)
    
    if executive_rows:
        df_exec = pd.DataFrame(executive_rows)
        print(df_exec.to_string(index=False))
        
        # Save executive table as CSV
        reports_dir = get_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "executive_validation_table.csv"
        df_exec.to_csv(csv_path, index=False)
        print(f"\n✅ Executive table saved to {csv_path}")
    else:
        print("  No data available for executive table.")
    
    # ========================================================================
    # 3. HEATMAP
    # ========================================================================
    print("\n📊 Generating Pearson Correlation Heatmap...")
    plot_pearson_heatmap(pearson_matrix, target_metrics)
    
    # ========================================================================
    # 4. SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("📋 VALIDATION SUMMARY")
    print("="*80)
    
    # Check if all ρ > 0.75
    all_pearson_above_threshold = True
    min_pearson = 1.0
    
    for key, val in pearson_matrix.items():
        if abs(val['r']) < 0.75:
            all_pearson_above_threshold = False
        if abs(val['r']) < min_pearson:
            min_pearson = abs(val['r'])
    
    if all_pearson_above_threshold and len(pearson_matrix) > 0:
        print(f"✅ All ρ > 0.75 confirms pipeline convergence across ref/non-ref/self-val")
        print(f"   Minimum Pearson r: {min_pearson:.4f}")
    else:
        print(f"⚠️  Some correlations below 0.75 threshold")
        print(f"   Minimum Pearson r: {min_pearson:.4f}")
    
    results['summary'] = {
        'all_pearson_above_threshold': bool(all_pearson_above_threshold),
        'min_pearson': float(min_pearson) if len(pearson_matrix) > 0 else None,
        'ref_records': int(len(df_ref)),
        'non_ref_records': int(len(df_non)),
        'self_val_records': int(len(df_self))
    }
    
    print("="*80)
    
    return results

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def save_validation_report(results, output_path=None):
    """Save validation metrics to JSON file."""
    if results is None:
        return None
    
    if output_path is None:
        reports_dir = get_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "framework_validation_metrics.json"
    
    # Convert numpy types to Python native types before JSON serialization
    results_clean = convert_numpy_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\n✅ Validation report saved to {output_path}")
    return output_path

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def run_validation():
    """Run framework validation metrics and save report."""
    results = compute_validation_metrics()
    if results:
        save_validation_report(results)
    return results

def run_dashboard():
    setup_dirs()
    df_ref, df_non = load_data()
    
    if not df_ref.empty:
        plot_risk_waterfall(df_ref)
        plot_combined_radars(df_ref)
        
        # Print and save health problem metrics table
        print_health_problem_metrics_table(df_ref, metric='semantic_similarity')
        save_health_problem_metrics_table(df_ref, metric='semantic_similarity')
    
    if not df_non.empty:
        plot_production_matrix(df_non)
        plot_pareto_curve(df_non) # Now uses NON-REF
        plot_violin_risk(df_non)  # New
        
    if not df_ref.empty and not df_non.empty:
        plot_eval_validation(df_ref, df_non)
    
    # Run validation metrics
    print("\n" + "="*80)
    print("🔬 Running Framework Validation Metrics...")
    print("="*80)
    run_validation()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--validation-only':
        run_validation()
    else:
        run_dashboard()
