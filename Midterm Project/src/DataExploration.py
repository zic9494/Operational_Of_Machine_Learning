from pandas import DataFrame
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run(df :DataFrame):
    print("Dtypes:\n",df.dtypes, "\n")
    print("\nDataFrame Shape:", df.shape)

    missvalues = df.isnull().sum()
    missing_percentage = (missvalues / len(df) ) * 100
    print("Miss values:\n", missvalues)
    print("Miss values percentage:\n", missing_percentage)

    os.makedirs("Charts", exist_ok=True)
    
    numerical_features = df.select_dtypes(include="number")
    print("\nNumerical Feature Statistics:\n",numerical_features.describe())

    numerical_features.hist(figsize=(14, 10), bins=20, edgecolor="black")
    plt.suptitle("Histograms of Numerical Features")
    plt.tight_layout()
    output_path = "Charts/numerical_histograms.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nHistogram image saved to: {output_path}")

    log_cols = ["incentive", "idle_time", "idle_men", "no_of_style_change"]
    available_log_cols = [col for col in log_cols if col in df.columns]
    if not available_log_cols:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(available_log_cols):
        ax = axes[i]
        series = df[col].dropna()
        positive = series[series > 0]
        skipped_count = len(series) - len(positive)

        if positive.empty:
            ax.text(0.5, 0.5, "No positive values\n(log scale unavailable)", ha="center", va="center")
            ax.set_title(col)
            continue

        ax.boxplot(positive, vert=False)
        ax.set_xscale("log")
        ax.set_title(col)
        ax.set_xlabel("Value (log scale)")
        ax.set_ylabel("")
        if skipped_count > 0:
            ax.text(
                0.98, 0.95,
                f"Skipped <= 0: {skipped_count}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8
            )

    for j in range(len(available_log_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Selected Features Boxplots (Log X-axis)")
    fig.tight_layout()
    log_output_path = "Charts/selected_log_boxplots.png"
    fig.savefig(log_output_path, dpi=150)
    print(f"Log-scale boxplot image saved to: {log_output_path}")
