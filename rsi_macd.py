import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from matplotlib import pyplot as plt

def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Oblicza wykładniczą średnią kroczącą (EMA) o zadanym okresie.
    """
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(df: pd.DataFrame, close_col: str = 'close',
                 fast: int = 8, slow: int = 21, signal: int = 5) -> pd.DataFrame:

    ema_fast = ema(df[close_col], span=fast)
    ema_slow = ema(df[close_col], span=slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal)
    macd_hist = macd_line - signal_line
    
    df[f'MACD_{fast}_{slow}_{signal}'] = macd_line
    df[f'MACDs_{fast}_{slow}_{signal}'] = signal_line
    df[f'MACDh_{fast}_{slow}_{signal}'] = macd_hist
    
    return df


def simulate_trading(df, buy_confirmations, sell_confirmations, initial_wallet=1000, fee=0.0):
    """
    Symulacja strategii tradingowej z opcjonalną prowizją.
    
    fee: opłata transakcyjna jako procent, np. 0.001 dla 0.1%.
    """
    wallet = initial_wallet

    for i in range(len(sell_confirmations)):
        buy_price = df.iloc[int(buy_confirmations[i])].close
        sell_price = df.iloc[int(sell_confirmations[i])].close

        wallet = wallet / buy_price  # kupujemy BTC za USDT
        wallet *= (1 - fee)          # prowizja za kupno

        wallet = wallet * sell_price # sprzedajemy BTC za USDT
        wallet *= (1 - fee)          # prowizja za sprzedaż

    profit = wallet - initial_wallet
    profit_p = 100 * profit / initial_wallet

    return wallet, profit, profit_p


# === Wczytanie i przygotowanie danych ===
df = pd.read_pickle('./data/binance-BTCUSDT-1h.pkl')
df_train = df[df.index <= '2023-12-31 23:00:00'].copy()
df_test = df[df.index > '2023-12-31 23:00:00'].copy()

df = df_test.copy()

# Upewnij się, że indeks jest kolumną 'date'
df = df.reset_index()
df['date'] = pd.to_datetime(df['date_open'])
df.set_index('date', inplace=False)


df['RSI_10'] = RSIIndicator(df['close'], window=10).rsi()
df['MACD_8_21_5'] = MACD(close=df['close'], window_fast=8, window_slow=21, window_sign=5).macd()
df = compute_macd(df, close_col='close', fast=8, slow=21, signal=5)

# === Logika strategii ===
buy_signals = np.array([])
buy_confirmations = np.array([])
sell_signals = np.array([])
sell_confirmations = np.array([])

buy_confirmation = False
buy_signal = False
sell_signal = False
RSI_below_30 = False

print('\n' + 10*'-' + "Operations Records" + 10*'-')

for i in range(len(df)):
    if df.RSI_10[i] < 30.0 and not RSI_below_30 and not buy_confirmation:
        RSI_below_30 = True

    if df.RSI_10[i] > 30.0 and RSI_below_30:
        buy_signal = True
        i_buy_signal = i
        buy_signals = np.append(buy_signals, i)

    if buy_signal:
        for j in range(len(df) - i):
            index = i_buy_signal + j
            if index >= len(df):
                break
            if df.MACD_8_21_5.iloc[index] > df.MACDs_8_21_5.iloc[index]:
                print('BUY  -  %s  -  BTC: Close price [USDT]: %.2f' % (df.iloc[i].date, df.iloc[i].close))
                buy_confirmations = np.append(buy_confirmations, index)
                buy_confirmation = True
                break
        buy_signal = False
        RSI_below_30 = False

    if df.RSI_10[i] >= 70 and not sell_signal and buy_confirmation:
        sell_signal = True
        i_sell_signal = i
        sell_signals = np.append(sell_signals, i)

    if sell_signal:
        for j in range(len(df) - i):
            index = i_sell_signal + j
            if index >= len(df):
                break
            if df.MACD_8_21_5.iloc[index] < df.MACDs_8_21_5.iloc[index]:
                print('SELL -  %s  -  BTC: Close price [USDT]: %.2f' % (df.iloc[i].date, df.iloc[i].close))
                sell_confirmations = np.append(sell_confirmations, index)
                buy_confirmation = False
                break
        sell_signal = False

# === Wyniki ===
print('\n' + 10*'-' + ' Results without fees ' + 10*'-')
wallet_no_fee, profit_no_fee, profit_p_no_fee = simulate_trading(df, buy_confirmations, sell_confirmations, initial_wallet=1000, fee=0.0)

print(f"Initial Wallet: USDT 1000.00")
print(f"Post-operations Wallet: USDT {wallet_no_fee:.2f}")
print(f"Profit: USDT {profit_no_fee:.2f}")
print(f"Cumulative returns: {profit_p_no_fee:.2f}%")

print('\n' + 10*'-' + ' Results with 0.1% fees ' + 10*'-')
wallet_with_fee, profit_with_fee, profit_p_with_fee = simulate_trading(df, buy_confirmations, sell_confirmations, initial_wallet=1000, fee=0.001)

print(f"Initial Wallet: USDT 1000.00")
print(f"Post-operations Wallet: USDT {wallet_with_fee:.2f}")
print(f"Profit: USDT {profit_with_fee:.2f}")
print(f"Cumulative returns: {profit_p_with_fee:.2f}%")

# === Statystyki liczby transakcji ===
buy_count = len(buy_confirmations)
sell_count = len(sell_confirmations)

print(f"\nLiczba sygnałów KUPNA  (BUY):  {buy_count}")
print(f"Liczba sygnałów SPRZEDAŻY (SELL): {sell_count}")

# === Wykresy ===
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.title('Buy and Sell Signals')
plt.ylabel('RSI')
plt.plot(df.date, df.RSI_10, label='RSI', zorder=1)
plt.scatter(df.date[buy_signals.astype(int)], df.RSI_10[buy_signals.astype(int)], color='g', marker='x', zorder=2)
plt.scatter(df.date[sell_signals.astype(int)], df.RSI_10[sell_signals.astype(int)], color='r', marker='x', zorder=2)
plt.axhline(30, color='green', linestyle='--')
plt.axhline(70, color='red', linestyle='--')
plt.legend()
plt.xlim([df.date.iloc[0], df.date.iloc[-1]])

plt.subplot(2, 1, 2)
plt.title('Buy and Sell Confirmations')
plt.ylabel("MACD")
plt.plot(df.date, df.MACD_8_21_5, label='MACD line')
plt.plot(df.date, df.MACDs_8_21_5, label='MACD signal')
plt.scatter(df.date[buy_confirmations.astype(int)], df.MACD_8_21_5[buy_confirmations.astype(int)], color='g', marker='x', zorder=2)
plt.scatter(df.date[sell_confirmations.astype(int)], df.MACD_8_21_5[sell_confirmations.astype(int)], color='r', marker='x', zorder=2)
plt.legend()
plt.xlim([df.date.iloc[0], df.date.iloc[-1]])

plt.tight_layout()
plt.show()

