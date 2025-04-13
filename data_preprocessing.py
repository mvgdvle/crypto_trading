import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator, MFIIndicator
from sklearn.preprocessing import StandardScaler
import talib


def prepare_data(symbols, timeframe='1h', split_data='2023-12-31 23:00:00'):
    df = pd.read_pickle('./data/binance-' + symbols.replace('/', '') + '-' + timeframe + '.pkl')

    df['feature_close'] = df['close'].pct_change()
    df['feature_high'] = df['high'] / df['close']
    df['feature_low'] = df['low'] / df['close']
    df['feature_volume'] = df['volume'] / df['volume'].rolling(7*24).max()

    # technical indicators
    df['feature_rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['feature_macd'] = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9).macd()
    df['feature_mfi'] = MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).money_flow_index()
    df['feature_atr'] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df['feature_natr'] = (df['feature_atr'] / df['close']) * 100
    df['feature_co'] = ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]).chaikin_money_flow()
    df['feature_obv'] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    df['feature_mom'] = df['close'].diff(periods=12)
    df['feature_htdcphase'] = talib.HT_DCPHASE(df['close'])
    df['feature_htsine'] = talib.HT_SINE(df['close'])[0]
    df['feature_httmm'] = talib.HT_TRENDMODE(df['close'])


    features_to_normalize = [
        'feature_close', 'feature_high', 'feature_low', 'feature_volume',
        'feature_macd', 'feature_rsi', 'feature_mfi', 'feature_atr',
        'feature_natr', 'feature_co', 'feature_obv',
        'feature_mom', 'feature_htdcphase', 'feature_htsine', 'feature_httmm']

    df.dropna(inplace=True)

    df_train = df[df.index <= split_data].copy()
    df_test = df[df.index > split_data].copy()

    df_train[features_to_normalize] = df_train[features_to_normalize].astype(np.float64)
    df_test[features_to_normalize] = df_test[features_to_normalize].astype(np.float64)

    scaler = StandardScaler()
    df_train.loc[:, features_to_normalize] = scaler.fit_transform(df_train[features_to_normalize])
    df_test.loc[:, features_to_normalize] = scaler.transform(df_test[features_to_normalize])

    return [df_train, df_test, scaler, features_to_normalize]

