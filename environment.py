import gymnasium as gym
import gym_trading_env
import numpy as np
from gym_trading_env.environments import basic_reward_function


def create_env(df, initial_position='random', windows=None, trading_fees=0, reward_function=basic_reward_function):
    env = gym.make('TradingEnv',
            df = df, 
            positions = [0, 1],
            initial_position = initial_position,
            windows = windows,
            trading_fees = trading_fees, 
            reward_function = reward_function
        )
    
    env.reset(seed=2025)

    env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0))
    env.add_metric('Episode Length', lambda history : len(history['position']))
    env.add_metric('Final Balance', lambda history: history['portfolio_valuation'][-1])
    env.add_metric('Total Reward', lambda history: np.sum(history['reward']))

    def returns(history):
        if len(history['portfolio_valuation']) < 2:
            return np.array([0])
        return np.diff(history['portfolio_valuation']) / history['portfolio_valuation'][:-1]
    
    env.add_metric('Mean Return', lambda history: np.mean(returns(history)) if len(returns(history)) > 0 else 0)
    env.add_metric('Win Rate', lambda history: np.sum(returns(history) > 0) / len(returns(history)) 
                   if len(returns(history)) > 0 else 0)
    env.add_metric('Volatility', lambda history: np.std(returns(history)) if len(returns(history)) > 0 else 0)
    env.add_metric('Downside Deviation', lambda history: np.std(returns(history)[returns(history) < 0]) 
                   if len(returns(history)[returns(history) < 0]) > 0 else 0)
    env.add_metric('Max Drawdown', lambda history: np.max(np.maximum.accumulate(history['portfolio_valuation']) - history['portfolio_valuation']) 
                   if len(history['portfolio_valuation']) > 0 else 0)
    env.add_metric('Sharpe Ratio', lambda history: np.mean(returns(history)) / np.std(returns(history)) 
                   if len(returns(history)) > 0 and np.std(returns(history)) != 0 else 0)
    env.add_metric('Sortino Ratio', lambda history: np.mean(returns(history)) / np.std(returns(history)[returns(history) < 0]) 
                   if len(returns(history)[returns(history) < 0]) > 0 and np.std(returns(history)[returns(history) < 0]) != 0 else 0)
    env.add_metric('Calmar Ratio', lambda h: np.mean(returns(h)) / np.max(np.maximum.accumulate(h['portfolio_valuation']) - h['portfolio_valuation']) 
                   if len(returns(h)) > 0 and np.max(np.maximum.accumulate(h['portfolio_valuation']) - h['portfolio_valuation']) != 0 else 0)


    # Avg Profit / Avg Loss
    def profit_loss_ratio(history):
        r = returns(history)
        profits = r[r > 0]
        losses = r[r < 0]
        avg_p = np.mean(profits) if len(profits) > 0 else 0
        avg_l = np.abs(np.mean(losses)) if len(losses) > 0 else 1e-9  
        return avg_p / avg_l
    env.add_metric('Avg Profit / Avg Loss', profit_loss_ratio)

    
    return env
