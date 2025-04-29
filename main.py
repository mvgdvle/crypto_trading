from data_preprocessing import prepare_data
from parameter_optimization import run_all_optimizations
from gym_trading_env.environments import basic_reward_function
from training import train
from test import test_model
from stable_baselines3 import PPO, DQN, A2C
from utils import set_seed

set_seed(2025)


data_var = {'symbols': ['BTC/USDT', 'ETH/USDT'], 'timeframe': '1h', 'split_data': '2023-12-31 23:00:00'} 
env_var = {'initial_position': ['random', 0], 'windows': [12, 50, 72], 'trading_fees': [0, 0.1/100], 'reward_function': basic_reward_function}
train_var = {'policy': ["MlpPolicy"], 'timesteps': 1000000}


# Bitcoin df
df_btc = prepare_data(symbols = data_var['symbols'][0], timeframe = data_var['timeframe'], split_data = data_var['split_data'])
df_train_btc = df_btc[0]
df_test_btc = df_btc[1]


if __name__ == "__main__":
##--------------------------- MODELS WITHOUT FEE ------------------------------
  # # PPO - without trading fee, window 50, default hyperparameters
  # train(model_class = PPO,
  #       algorithm_name = "PPO",
  #       policy = train_var['policy'][0],
  #       df = df_train_btc, 
  #       dir = 'basic/no_fee/PPO/PPO50',
  #       initial_position = env_var['initial_position'][0], 
  #       windows = env_var['windows'][1], 
  #       trading_fees = env_var['trading_fees'][0], 
  #       reward_function = env_var['reward_function'], 
  #       timesteps = train_var['timesteps'])

  # test_model(df = df_test_btc,
  #           model = PPO,
  #           dir = 'basic/no_fee/PPO/PPO50', 
  #           initial_position = env_var['initial_position'][1], 
  #           windows = env_var['windows'][1],
  #           trading_fees = env_var['trading_fees'][0], 
  #           reward_function = env_var['reward_function'])
  
  # A2C - without trading fee, window 50, default hyperparameters
#   train(model_class = A2C,
#         algorithm_name = "A2C",
#         policy = train_var['policy'][0],
#         df = df_train_btc, 
#         dir = 'basic/no_fee/A2C/A2C50_atr_v3',
#         initial_position = env_var['initial_position'][0], 
#         windows = env_var['windows'][1], 
#         trading_fees = env_var['trading_fees'][0], 
#         reward_function = env_var['reward_function'], 
#         timesteps = train_var['timesteps'])

  # test_model(df = df_test_btc,
  #           model = A2C,
  #           dir = 'basic/no_fee/A2C/A2C50_atr_v3_2', 
  #           initial_position = env_var['initial_position'][1], 
  #           windows = env_var['windows'][1],
  #           trading_fees = env_var['trading_fees'][0], 
  #           reward_function = env_var['reward_function'])
  
  # DQN - without trading fee, window 50, default hyperparameters
  train(model_class = DQN,
        algorithm_name = "DQN",
        policy = train_var['policy'][0],
        df = df_train_btc, 
        dir = 'basic/no_fee/DQN/DQN50',
        initial_position = env_var['initial_position'][0], 
        windows = env_var['windows'][1], 
        trading_fees = env_var['trading_fees'][0], 
        reward_function = env_var['reward_function'], 
        timesteps = train_var['timesteps'])

  test_model(df = df_test_btc,
            model = DQN,
            dir = 'basic/no_fee/DQN/DQN50', 
            initial_position = env_var['initial_position'][1], 
            windows = env_var['windows'][1],
            trading_fees = env_var['trading_fees'][0], 
            reward_function = env_var['reward_function'])
  

# # #--------------------------- MODELS WITH FEE ------------------------------
# # PPO - with trading fee, window 50, default hyperparameters
#   train(model_class = PPO,
#         algorithm_name = "PPO",
#         policy = train_var['policy'][0],
#         df = df_train_btc, 
#         dir = 'basic/with_fee/PPO/PPO50',
#         initial_position = env_var['initial_position'][0], 
#         windows = env_var['windows'][1], 
#         trading_fees = env_var['trading_fees'][1], 
#         reward_function = env_var['reward_function'], 
#         timesteps = train_var['timesteps'])

#   test_model(df = df_test_btc,
#             model = PPO,
#             dir = 'basic/with_fee/PPO/PPO50', 
#             initial_position = env_var['initial_position'][1], 
#             windows = env_var['windows'][1],
#             trading_fees = env_var['trading_fees'][1], 
#             reward_function = env_var['reward_function'])
  
#   # A2C - with trading fee, window 50, default hyperparameters
#   train(model_class = A2C,
#         algorithm_name = "A2C",
#         policy = train_var['policy'][0],
#         df = df_train_btc, 
#         dir = 'basic/with_fee/A2C/A2C50',
#         initial_position = env_var['initial_position'][0], 
#         windows = env_var['windows'][1], 
#         trading_fees = env_var['trading_fees'][1], 
#         reward_function = env_var['reward_function'], 
#         timesteps = train_var['timesteps'])

#   test_model(df = df_test_btc,
#             model = A2C,
#             dir = 'basic/with_fee/A2C/A2C50', 
#             initial_position = env_var['initial_position'][1], 
#             windows = env_var['windows'][1],
#             trading_fees = env_var['trading_fees'][1], 
#             reward_function = env_var['reward_function'])
  
#   # DQN - with trading fee, window 50, default hyperparameters
#   train(model_class = DQN,
#         algorithm_name = "DQN",
#         policy = train_var['policy'][0],
#         df = df_train_btc, 
#         dir = 'basic/with_fee/DQN/DQN50',
#         initial_position = env_var['initial_position'][0], 
#         windows = env_var['windows'][1], 
#         trading_fees = env_var['trading_fees'][1], 
#         reward_function = env_var['reward_function'], 
#         timesteps = train_var['timesteps'])

#   test_model(df = df_test_btc,
#             model = DQN,
#             dir = 'basic/with_fee/DQN/DQN50', 
#             initial_position = env_var['initial_position'][1], 
#             windows = env_var['windows'][1],
#             trading_fees = env_var['trading_fees'][1], 
#             reward_function = env_var['reward_function'])

