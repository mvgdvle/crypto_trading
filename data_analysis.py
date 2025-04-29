import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator, MFIIndicator
from sklearn.preprocessing import StandardScaler
import talib
import os


df = pd.read_pickle('./data/binance-BTCUSDT-1h.pkl')

df.isnull().sum()

df['feature_close'] = df['close'].pct_change()      # ( close[t] - close[t-1] )/ close[t-1]
df['feature_open'] = df['open']/df['close']
df['feature_high'] = df['high']/df['close']
df['feature_low'] = df['low']/df['close'] 
df['feature_volume'] = df['volume'] / df['volume'].rolling(7*24).max()      # volume[t] / max(*volume[t-7*24:t+1])

# braki przed dodatkowymi zmiennymi
df.isnull().sum()


# technical indicators
df['feature_rsi'] = RSIIndicator(df['close'], window=14).rsi()
df['feature_macd'] = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9).macd()
df['feature_mfi'] = MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).money_flow_index()
df['feature_atr'] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
df['feature_obv'] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
df['feature_mom'] = df['close'].diff(periods=12)
df['feature_htdcphase'] = talib.HT_DCPHASE(df['close'])
df['feature_htsine'] = talib.HT_SINE(df['close'])[0]
df['feature_httcm'] = talib.HT_TRENDMODE(df['close'])

# braki po dodatkowych zmiennych
df.isnull().sum()
df.dropna(inplace=True)
features = [col for col in df.columns if col.startswith('feature_')]
df_raw = df.copy()

# Normalizacja
scaler = StandardScaler()
df_norm = df.copy()
df_norm[features] = scaler.fit_transform(df_norm[features])
missing = df_norm[features].isnull().sum()

# Wartości odstające
outliers_raw = ((np.abs(df[features] - df[features].mean()) > 3 * df[features].std())).sum()
outliers_norm = ((np.abs(df_norm[features]) > 3)).sum()

# Statystyki
stats = df[features].describe().T
corr = df[features].corr()


os.makedirs("data/analysis", exist_ok=True)

# for col in features:
#     plt.figure(figsize=(10, 4))
#     sns.histplot(df[col].dropna(), bins=100, kde=True)
#     plt.title(f'Histogram: {col}')
#     plt.savefig(f'data/analysis/hist_{col}.png')
#     plt.close()

#     plt.figure(figsize=(10, 2))
#     sns.boxplot(x=df[col].dropna())
#     plt.title(f'Boxplot: {col}')
#     plt.savefig(f'data/analysis/box_{col}.png')
#     plt.close()


# # Wykres cen + RSI
# plt.figure(figsize=(15, 6))
# ax1 = df['feature_rsi'].plot(label='RSI', secondary_y=True, color='green')
# ax2 = df['close'].plot(label='Cena zamknięcia', color='black')
# plt.title("Cena Bitcoina i RSI")
# ax1.set_ylabel("RSI")
# ax2.set_ylabel("Cena zamknięcia")
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper left')
# plt.savefig("data/analysis/close_rsi.png")
# plt.close()


# # Wykres ceny + OBV
# plt.figure(figsize=(15, 6))
# ax1 = df['close'].plot(label='Close Price', color='black')
# ax2 = df['feature_obv'].plot(label='OBV', secondary_y=True, color='purple')
# plt.title("BTC Price and OBV")
# ax1.set_ylabel("Close Price")
# ax2.set_ylabel("OBV")
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# plt.savefig("data/analysis/close_obv.png")
# plt.close()

# Heatmapa korelacji
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap='RdBu',         
    center=0,
    annot_kws={"size": 12},            
    cbar_kws={'label': 'Współczynnik korelacji'}
)
plt.title("Macierz korelacji zmiennych", fontsize=20)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("data/analysis/correlation_matrix.png")
plt.close()


# # Zapis danych statystycznych i braków
# stats.to_csv("data/analysis/stats.csv")
# pd.DataFrame({'Missing (norm)': missing, 'Outliers (raw)': outliers_raw, 'Outliers (norm)': outliers_norm}).to_csv("data/analysis/missing_outliers.csv")





