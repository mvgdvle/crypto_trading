import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import time
import json
import os
from gym_trading_env.environments import basic_reward_function

from environment import create_env


def train(model_class, algorithm_name, policy, df, dir, timesteps=100_000,
                initial_position='random', windows=None, trading_fees=0,
                reward_function=basic_reward_function, params_path=None):
    """
    Trenuje wskazany model RL (PPO, A2C, DQN) i zapisuje metryki.

    Parametry:
    - model_class: PPO / DQN / A2C z stable_baselines3
    - algorithm_name: str – "PPO", "DQN", "A2C"
    - df: dane treningowe
    - dir: katalog/model_id
    - timesteps: liczba kroków uczenia
    - params_path: (opcjonalne) ścieżka do pliku .json z parametrami
    """

    env = DummyVecEnv([lambda: create_env(df, initial_position, windows, trading_fees, reward_function)])

    custom_params = {}
    if params_path and os.path.exists(params_path):
        with open(params_path, "r") as f:
            all_params = json.load(f)
            if algorithm_name in all_params:
                custom_params = all_params[algorithm_name]
                print(f"Loaded {algorithm_name} params from {params_path}")
            else:
                print(f"No section '{algorithm_name}' in {params_path}. Using defaults.")
    elif params_path:
        print(f"File '{params_path}' not found. Using defaults.")

    # if algorithm_name == "DQN" and "buffer_size" not in custom_params:
    #     custom_params["buffer_size"] = 5000

    model = model_class(policy, env, verbose=1, **custom_params)

    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    end_time = time.time()
    train_duration = end_time - start_time

    model_path = f"models/{dir}"
    catalog_model = model_path.rsplit('/', 1)[0]
    os.makedirs(catalog_model, exist_ok=True)
    model.save(model_path)

    log_data = model.logger.name_to_value if hasattr(model.logger, 'name_to_value') else {}

    training_info = {
        "algorithm": algorithm_name,
        "timesteps": timesteps,
        "train_duration_sec": round(train_duration, 2),
        "train_duration_min": round(train_duration / 60, 2),
        "windows": windows,
        "trading_fees": trading_fees,
        "model_path": model_path,
        "value_loss": log_data.get("train/value_loss", None),
        "entropy_loss": log_data.get("train/entropy_loss", None),
        "used_custom_params": bool(custom_params),
        "custom_params_file": params_path if custom_params else None
    }

    training_path = f"training_logs/{dir}"
    catalog_training = training_path.rsplit('/', 1)[0]
    os.makedirs(catalog_training, exist_ok=True)
    with open(f"training_logs/{dir}.json", "w") as f:
        json.dump(training_info, f, indent=4)

    print(f"✔ {algorithm_name} trained in {training_info['train_duration_min']} min. "
          f"Info saved to training_logs/{dir}.json")

