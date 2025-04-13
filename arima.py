import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from environment import create_env
from itertools import product
import json
import warnings
import os
warnings.filterwarnings("ignore")


def prepare_data_arima(symbol="BTC/USDT", timeframe='1h', split_data='2023-12-31 23:00:00'):
    """
    Przygotowuje dane specjalnie dla modelu ARIMA:
    - tylko kolumna 'close'
    - bez normalizacji
    - bez dodatkowych zmiennych technicznych
    """
    df = pd.read_pickle(f'./data/binance-{symbol.replace("/", "")}-{timeframe}.pkl')
    df = df[['close']].copy().dropna()
    df_train = df[df.index <= split_data].copy()
    df_test = df[df.index > split_data].copy()
    return df_train, df_test


def tune_arima_params(series, p_range=(0, 3), d_range=(0, 1), q_range=(0, 3)):
    best_aic = float("inf")
    best_order = None

    for order in product(range(p_range[0], p_range[1]+1),
                         range(d_range[0], d_range[1]+1),
                         range(q_range[0], q_range[1]+1)):
        try:
            model = ARIMA(series, order=order)
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
        except:
            continue
    return best_order


def create_arima_strategy(df_train, df_test, horizon=1, order=None):
    df_all = pd.concat([df_train, df_test])
    df_all = df_all[["close"]].copy()

    preds = []
    for t in range(len(df_train), len(df_all) - horizon):
        train_series = df_all["close"].iloc[:t]
        try:
            model = ARIMA(train_series, order=order)
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=horizon)[-1]
            preds.append(yhat)
        except:
            preds.append(np.nan)

    df_test = df_test.iloc[horizon:].copy()
    df_test["predicted_price"] = preds

    df_test["signal"] = 0
    df_test.loc[df_test["predicted_price"] > df_test["close"], "signal"] = 1
    df_test["position"] = df_test["signal"].shift(1).fillna(0)

    df_test["portfolio_valuation"] = (1 + df_test["position"] * df_test["close"].pct_change().fillna(0)).cumprod()
    df_test["reward"] = df_test["portfolio_valuation"].pct_change().fillna(0)

    return df_test


def evaluate_strategy(df_test):
    env = create_env(df_test)
    history = {
        "position": df_test["position"].values,
        "portfolio_valuation": df_test["portfolio_valuation"].values,
        "reward": df_test["reward"].values
    }
    return env.get_metrics_from_history(history)


def convert_np_types(obj):
    if isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    model_name = "ARIMA"
    model_id = "classic/arima_tuned"
    os.makedirs("test_logs", exist_ok=True)

    df_train, df_test = prepare_data_arima("BTC/USDT")

    best_order = tune_arima_params(df_train["close"])
    print(f"Najlepsze parametry ARIMA: {best_order}")

    print("Tworzenie strategii i predykcji...")
    result_df = create_arima_strategy(df_train, df_test, order=best_order)
    metrics = evaluate_strategy(result_df)

    test_result = {
        "model": model_name,
        "order": best_order,
        "timesteps": len(result_df),
    }
    test_result.update(metrics)

    with open(f"test_logs/{model_id}.json", "w") as f:
        json.dump(test_result, f, indent=4, default=convert_np_types)

    print(f"Wyniki testu zapisane w test_logs/{model_id}.json")
    for k, v in metrics.items():
        print(f"{k}: {v}")
