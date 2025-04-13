import optuna
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import json
import os
from environment import create_env


def make_env(df, initial_position, windows, trading_fees, reward_function):
    def _init():
        env = create_env(df, initial_position, windows, trading_fees, reward_function)
        env = Monitor(env) 
        return env
    return DummyVecEnv([_init])


# Hyperparameters space
def sample_ppo_params(trial):
    return {
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0)
    }

def sample_a2c_params(trial):
    return {
        "n_steps": trial.suggest_categorical("n_steps", [5, 16, 32]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1)
    }

def sample_dqn_params(trial):
    return {
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    }


def optimize_model(policy, algorithm_class, param_sampler, df, env_params, total_timesteps=50000, n_trials=3):
    def objective(trial):
        params = param_sampler(trial)
        env = make_env(df, **env_params)

        model = algorithm_class(policy, env, verbose=0, **params)
        model.learn(total_timesteps=total_timesteps)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        return mean_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study


def run_all_optimizations(policy, df, dir, env_params, total_timesteps=50000, n_trials=3):
    results = {
        "PPO": optimize_model(policy, PPO, sample_ppo_params, df, env_params, total_timesteps, n_trials).best_params,
        "A2C": optimize_model(policy, A2C, sample_a2c_params, df, env_params, total_timesteps, n_trials).best_params,
        "DQN": optimize_model(policy, DQN, sample_dqn_params, df, env_params, total_timesteps, n_trials).best_params
    }

    
    params_path = f"models/params/{dir}"
    catalog_params = params_path.rsplit('/', 1)[0]
    os.makedirs(catalog_params, exist_ok=True)
    with open(f"models/params/{dir}", 'w') as f:
        json.dump(results, f, indent=4)

    return results

