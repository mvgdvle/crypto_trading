import json
import numpy as np
import pandas as pd
import os
from environment import create_env


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
    

def test_model(df, model, dir, initial_position, windows, trading_fees, reward_function):
    env_test = create_env(
        df, 
        initial_position=initial_position, 
        windows=windows, 
        trading_fees=trading_fees, 
        reward_function=reward_function)

    model = model.load('models/' + dir)

    done, truncated = False, False
    observation, info = env_test.reset()
    while not done and not truncated:
        action, _states = model.predict(observation)
        observation, reward, done, truncated, info = env_test.step(action)
        env_test.save_for_render(dir = 'render_logs/' + dir)


    test_metrics = env_test.get_metrics()
    test_results = {
        "model": dir,
        "windows": windows,
        "initial_position": initial_position,
        "trading_fees": trading_fees
    }

    last_info = {"last_info": info}

    test_results.update(test_metrics)
    test_results.update(last_info)

    test_path = f"test_logs/{dir}"
    catalog_test = test_path.rsplit('/', 1)[0]
    os.makedirs(catalog_test, exist_ok=True)
    with open(f"test_logs/{dir}.json", "w") as f:
        json.dump(test_results, f, indent=4, default=convert_np_types)

    print(f"✔ Wyniki testu modelu {dir} zapisane w test_logs/{dir}.json")

