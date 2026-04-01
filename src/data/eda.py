"""
NetShield — Step 2: Exploratory Data Analysis
Run from the netshield/ project root:
    python -m src.data.eda

Or copy into a Jupyter notebook for interactive exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Configuration ───
RAW_DATA_DIR = Path("data/raw")
CSV_FILE = RAW_DATA_DIR / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

FIGURE_DIR = Path("notebooks/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV with basic type inference."""
    print(f"Loading {path.name}...")
    df = pd.read_csv(path, low_memory=False)

    # Strip whitespace from column names (known issue with this dataset)
    df.columns = df.columns.str.strip()

    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def inspect_basics(df: pd.DataFrame) -> None:
    """Print basic info about the dataframe."""
    print("\n")
    print("BASIC INFO")

    print(f"\nShape: {df.shape}")
    print(f"\nColumn dtypes:")
    print(df.dtypes.value_counts().to_string())

    print(f"\nFirst 5 column names: {df.columns[:5].tolist()}")
    print(f"Last 5 column names:  {df.columns[-5:].tolist()}")

    # Check for the Label column
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    if label_col:
        print(f"\n Label column found: '{label_col}'")
        print(f"   Unique labels: {df[label_col].nunique()}")
        print(f"   Labels: {df[label_col].unique().tolist()}")
    else:
        print("\n  No 'Label' column found!")
        print(f"   All columns: {df.columns.tolist()}")


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing and infinite values."""
    print("\n" )
    print("MISSING & INFINITE VALUES")

    # NaN counts
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        print(f"\n⚠️  {len(nan_cols)} columns have NaN values:")
        for col, count in nan_cols.items():
            pct = count / len(df) * 100
            print(f"   {col}: {count:,} ({pct:.2f}%)")
    else:
        print("\nNo NaN values found")

    # Infinite values (only in numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    inf_cols = inf_counts[inf_counts > 0]

    if len(inf_cols) > 0:
        print(f"\n {len(inf_cols)} columns have infinite values:")
        for col, count in inf_cols.items():
            pct = count / len(df) * 100
            print(f"   {col}: {count:,} ({pct:.2f}%)")
    else:
        print("\n No infinite values found")

    # Negative values where they shouldn't exist
    # (packet counts, byte counts, durations should be >= 0)
    suspect_negative_cols = [
        c for c in numeric_df.columns
        if any(kw in c.lower() for kw in ["packet", "byte", "length", "duration", "size", "count"])
    ]
    print(f"\nChecking {len(suspect_negative_cols)} columns for unexpected negatives...")
    neg_found = False
    for col in suspect_negative_cols:
        neg_count = (numeric_df[col] < 0).sum()
        if neg_count > 0:
            neg_found = True
            print(f"     {col}: {neg_count:,} negative values")
    if not neg_found:
        print("    No unexpected negative values")

    return nan_cols


def analyze_class_distribution(df: pd.DataFrame) -> None:
    """Analyze the label distribution."""
    print("\n")
    print("CLASS DISTRIBUTION")


    label_col = "Label"
    if label_col not in df.columns:
        # Try to find it case-insensitively
        for col in df.columns:
            if col.lower() == "label":
                label_col = col
                break

    counts = df[label_col].value_counts()
    pcts = df[label_col].value_counts(normalize=True) * 100

    print(f"\n{'Class':<30} {'Count':>10} {'Percent':>10}")
    print("-" * 52)
    for label in counts.index:
        print(f"{label:<30} {counts[label]:>10,} {pcts[label]:>9.2f}%")
    print("-" * 52)
    print(f"{'TOTAL':<30} {counts.sum():>10,}")

    benign_pct = pcts.get("Benign", 0)
    attack_pct = 100 - benign_pct
    print(f"\nBenign: {benign_pct:.1f}% | Attack: {attack_pct:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    counts.plot(kind="barh", ax=axes[0], color=sns.color_palette("husl", len(counts)))
    axes[0].set_title("Class Distribution (Count)")
    axes[0].set_xlabel("Number of Flows")
    axes[0].invert_yaxis()

    # Log-scale bar chart (to see minority classes)
    counts.plot(kind="barh", ax=axes[1], color=sns.color_palette("husl", len(counts)), log=True)
    axes[1].set_title("Class Distribution (Log Scale)")
    axes[1].set_xlabel("Number of Flows (log)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {FIGURE_DIR / 'class_distribution.png'}")


def analyze_feature_distributions(df: pd.DataFrame) -> None:
    """Analyze distributions of numeric features."""
    print("\n" + "=" * 60)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 60)

    numeric_df = df.select_dtypes(include=[np.number])
    print(f"\nNumeric features: {numeric_df.shape[1]}")

    # Summary statistics
    stats = numeric_df.describe().T
    stats["skewness"] = numeric_df.skew()
    stats["kurtosis"] = numeric_df.kurtosis()

    # Flag highly skewed features
    highly_skewed = stats[stats["skewness"].abs() > 5]
    print(f"Highly skewed features (|skew| > 5): {len(highly_skewed)}")
    if len(highly_skewed) > 0:
        print(highly_skewed[["mean", "std", "min", "max", "skewness"]].to_string())

    # Features with zero variance (useless)
    zero_var = stats[stats["std"] == 0]
    if len(zero_var) > 0:
        print(f"\n⚠️  Zero-variance features (should drop): {zero_var.index.tolist()}")
    else:
        print("\n No zero-variance features")

    # Plot histograms for a selection of key features
    key_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Mean",
        "Bwd Packet Length Mean", "Flow IAT Mean", "SYN Flag Count",
        "Init_Win_bytes_forward"
    ]
    available = [f for f in key_features if f in numeric_df.columns]

    if available:
        n_cols = 3
        n_rows = (len(available) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, feat in enumerate(available):
            data = numeric_df[feat].replace([np.inf, -np.inf], np.nan).dropna()
            # Clip extreme outliers for visualization (99th percentile)
            upper = data.quantile(0.99)
            clipped = data[data <= upper]
            axes[i].hist(clipped, bins=50, edgecolor="black", alpha=0.7)
            axes[i].set_title(feat, fontsize=10)
            axes[i].set_ylabel("Count")

        # Hide empty subplots
        for i in range(len(available), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Key Feature Distributions (clipped at 99th percentile)", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {FIGURE_DIR / 'feature_distributions.png'}")


def analyze_correlations(df: pd.DataFrame) -> None:
    """Find highly correlated feature pairs (candidates for removal)."""
    print("\n")
    print("FEATURE CORRELATIONS")

    numeric_df = df.select_dtypes(include=[np.number])
    # Replace inf for correlation computation
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    corr = numeric_df.corr()

    # Find pairs with |correlation| > 0.95
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.95:
                high_corr_pairs.append(
                    (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                )

    print(f"\nHighly correlated pairs (|r| > 0.95): {len(high_corr_pairs)}")
    if high_corr_pairs:
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for f1, f2, r in high_corr_pairs[:20]:  # Show top 20
            print(f"   {r:+.3f}  {f1}  ↔  {f2}")

    # Heatmap of top features
    if len(numeric_df.columns) > 10:
        # Use a subset: pick features with highest variance
        top_var = numeric_df.var().nlargest(20).index
        sub_corr = numeric_df[top_var].corr()
    else:
        sub_corr = corr

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Top 20 by Variance)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {FIGURE_DIR / 'correlation_heatmap.png'}")


def analyze_attack_vs_benign(df: pd.DataFrame) -> None:
    """Compare feature distributions between benign and attack traffic."""
    print("\n")
    print("BENIGN vs ATTACK COMPARISON")

    label_col = "Label"
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    df["is_attack"] = df[label_col] != "Benign"

    key_features = [
        "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
        "Fwd Packet Length Mean", "Bwd Packet Length Mean",
        "SYN Flag Count"
    ]
    available = [f for f in key_features if f in df.columns]

    if available:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, feat in enumerate(available):
            data = df[[feat, "is_attack"]].copy()
            data[feat] = data[feat].replace([np.inf, -np.inf], np.nan)
            data = data.dropna()

            upper = data[feat].quantile(0.95)
            data = data[data[feat] <= upper]

            for label, group in data.groupby("is_attack"):
                tag = "Attack" if label else "Benign"
                axes[i].hist(group[feat], bins=50, alpha=0.5, label=tag, density=True)

            axes[i].set_title(feat, fontsize=10)
            axes[i].legend()

        for i in range(len(available), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Feature Distributions: Benign vs Attack (95th pctl)", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "benign_vs_attack.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {FIGURE_DIR / 'benign_vs_attack.png'}")

    # Drop temp column
    df.drop(columns=["is_attack"], inplace=True)


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of findings to guide preprocessing."""
    print("\n")
    print("SUMMARY — KEY FINDINGS FOR PREPROCESSING")

    numeric_df = df.select_dtypes(include=[np.number])

    nan_total = df.isna().sum().sum()
    inf_total = np.isinf(numeric_df).sum().sum()
    neg_cols = sum(
        1 for c in numeric_df.columns
        if any(kw in c.lower() for kw in ["packet", "byte", "length", "duration"])
        and (numeric_df[c] < 0).any()
    )
    zero_var_cols = (numeric_df.std() == 0).sum()

    print(f"""
Issues to handle in preprocessing:
  1. NaN values:          {nan_total:,} total across all columns
  2. Infinite values:     {inf_total:,} total
  3. Negative violations: {neg_cols} columns with unexpected negatives
  4. Zero-variance cols:  {zero_var_cols} (should drop)
  5. Class imbalance:     see distribution above
  6. High correlations:   see pairs above (candidates for removal)
  7. Skewed features:     most features are heavily right-skewed -> consider log transform

""")


# Main
if __name__ == "__main__":
    if not CSV_FILE.exists():
        print(f" File not found: {CSV_FILE}")
        print(f"   Make sure you've downloaded the dataset to {RAW_DATA_DIR}/")
        print(f"   See setup.sh for download instructions.")
        exit(1)

    df = load_data(CSV_FILE)
    inspect_basics(df)
    analyze_missing_values(df)
    analyze_class_distribution(df)
    analyze_feature_distributions(df)
    analyze_correlations(df)
    analyze_attack_vs_benign(df)
    print_summary(df)