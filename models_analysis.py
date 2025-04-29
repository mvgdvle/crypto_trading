import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path_str):
    wanted_keys = [
        "Market Return", "Portfolio Return", "Position Changes", "Episode Length",
        "Final Balance", "Total Reward", "Mean Return", "Win Rate", "Volatility",
        "Downside Deviation", "Max Drawdown", "Sharpe Ratio", "Sortino Ratio",
        "Calmar Ratio", "Avg Profit / Avg Loss"
    ]

    path = Path(path_str)
    rows = []

    for file in path.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)

        model_path = data.get("model", file.stem)
        algorithm = "A2C" if "a2c" in model_path.lower() else "PPO" if "ppo" in model_path.lower() else "DQN"
        fee_type = "no_fee" if "no_fee" in model_path.lower() else "with_fee"
        label = f"{algorithm}_{fee_type}"

        row = {"Model": label}
        for k in wanted_keys:
            value = data.get(k, None)
            if isinstance(value, str) and value.endswith('%'):
                value = float(value.strip('%'))
            row[k] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("Model", inplace=True)
    return df.transpose()



df_ppo_no_fee = load_data('test_logs/basic/no_fee/PPO')
df_a2c_no_fee = load_data('test_logs/basic/no_fee/A2C')
df_dqn_no_fee = load_data('test_logs/basic/no_fee/DQN')

df_ppo_with_fee = load_data('test_logs/basic/with_fee/PPO')
df_a2c_with_fee = load_data('test_logs/basic/with_fee/A2C')
df_dqn_with_fee = load_data('test_logs/basic/with_fee/DQN')


df_all = pd.concat([
    df_ppo_no_fee, df_a2c_no_fee, df_dqn_no_fee,
    df_ppo_with_fee, df_a2c_with_fee, df_dqn_with_fee
], axis=1)

# print(df_all)

# # Wybierz tylko jedną metrykę: Portfolio Return
# portfolio_return_df = df_all.loc[["Portfolio Return"]].transpose().reset_index()
# portfolio_return_df.columns = ["Model", "Portfolio Return"]

# # Wykres słupkowy porównujący Portfolio Return
# plt.figure(figsize=(10, 5))
# sns.barplot(data=portfolio_return_df, x="Model", y="Portfolio Return", palette="viridis")

# plt.title("Porównanie modeli – Portfolio Return (%)")
# plt.xticks(rotation=45)
# plt.ylabel("Portfolio Return (%)")
# plt.xlabel("Model")
# plt.tight_layout()
# plt.show()


# position_changes_df = df_all.loc[["Position Changes"]].transpose().reset_index()
# position_changes_df.columns = ["Model", "Position Changes"]

# # Wykres słupkowy porównujący Position Changes
# plt.figure(figsize=(10, 5))
# sns.barplot(data=position_changes_df, x="Model", y="Position Changes", palette="mako")

# plt.title("Porównanie modeli – Liczba zmian pozycji (Position Changes)")
# plt.xticks(rotation=45)
# plt.ylabel("Liczba zmian pozycji")
# plt.xlabel("Model")
# plt.tight_layout()
# plt.show()


# risk_metrics_df = df_all.loc[["Sharpe Ratio", "Sortino Ratio"]].copy()

# # Transpozycja: wiersze = modele, kolumna "Metric" + "Value"
# risk_metrics_df = risk_metrics_df.transpose().reset_index()
# risk_metrics_df = risk_metrics_df.rename(columns={"index": "Model"})

# # Zamiana na long format
# risk_metrics_long = risk_metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")

# # Wykres słupkowy
# plt.figure(figsize=(10, 5))
# sns.barplot(data=risk_metrics_long, x="Model", y="Value", hue="Metric")

# plt.title("Porównanie modeli – Sharpe Ratio i Sortino Ratio")
# plt.xticks(rotation=45)
# plt.ylabel("Wartość wskaźnika")
# plt.xlabel("Model")
# plt.legend(title="Metryka")
# plt.tight_layout()
# plt.show()

scatter_df = df_all.loc[["Portfolio Return", "Volatility"]].transpose().reset_index()
scatter_df.columns = ["Model", "Portfolio Return", "Volatility"]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=scatter_df, x="Volatility", y="Portfolio Return", hue="Model", s=120)
for i in range(len(scatter_df)):
    plt.text(scatter_df["Volatility"][i] + 0.00005, scatter_df["Portfolio Return"][i],
             scatter_df["Model"][i], fontsize=9)

plt.title("Modele: Zysk vs Ryzyko")
plt.xlabel("Volatility")
plt.ylabel("Portfolio Return (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
