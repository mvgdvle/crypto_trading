import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_results(log_dir="test_logs"):
    results = []
    for file in os.listdir(log_dir):
        if file.endswith(".json"):
            with open(os.path.join(log_dir, file), "r") as f:
                data = json.load(f)
                data["source_file"] = file
                results.append(data)
    return pd.DataFrame(results)

def plot_comparison(df, metrics, save_path="charts/model_comparison.png"):
    df["Model"] = df["model"].str.upper() + " - " + df["source_file"].str.replace(".json", "")
    melted_df = df[["Model"] + metrics].melt(id_vars="Model", var_name="Metric", value_name="Value")

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=melted_df,
        x="Metric", y="Value", hue="Model",
        kind="bar", height=6, aspect=2
    )
    g.set_xticklabels(rotation=45)
    g.figure.suptitle("Porównanie metryk modeli (RL vs ARIMA)", fontsize=16)
    plt.tight_layout()

    # Utwórz folder jeśli nie istnieje
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    g.figure.savefig(save_path)
    print(f"Wykres zapisany do pliku: {save_path}")

    plt.show()


if __name__ == "__main__":
    df_results = load_test_results("test_logs")

    metrics = [
        "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
        "Volatility", "Win Rate", "Mean Return", "Final Balance",
        "Calmar Ratio", "Avg Profit / Avg Loss"
    ]
    available_metrics = [m for m in metrics if m in df_results.columns]

    plot_comparison(df_results, available_metrics, save_path="charts/model_comparison.png")
