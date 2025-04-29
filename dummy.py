
# reward function

def pnl_sharpe_reward(history, fee=0, alpha=0.01):
    t = len(history['portfolio_valuation']) - 1
    if t < 1:
        return 0.0

    prices = history['portfolio_valuation']
    positions = history['position']

    # Zwrot portfela
    z_t = (prices[t] - prices[t - 1]) / prices[t - 1]

    # Pozycje: e_t ∈ [0, 1]
    e_t = positions[t]
    e_t_prev = positions[t - 1]

    # Nagroda PnL + kara za zmianę pozycji (prowizja)
    r_pnl = e_t * z_t
    r_fee = -fee * abs(e_t - e_t_prev)
    reward = r_pnl + r_fee

    # Sharpe
    m = len(prices)
    w = int(m * 0.5)
    if t >= w:
        returns = np.diff(prices[:t + 1]) / prices[:t]  # z0...zt
        mean_r = np.mean(returns)
        var_r = np.var(returns)

        r_sr = alpha * (mean_r / np.sqrt(var_r)) if var_r > 0 else 0.0
        reward += r_sr

    return reward


from sb3_contrib import RecurrentPPO
import sb3_contrib.common.recurrent.policies


import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")



  # #-------------------------- HYPERPARAMETERS OPTIMIZATION --------------------------------
  # # Models without fees
  # env_params_basic = {
  #   "initial_position": env_var['initial_position'][0],
  #   "windows": env_var['windows'][1],
  #   "trading_fees": env_var['trading_fees'][0],
  #   "reward_function": env_var['reward_function']
  #   }

  # run_all_optimizations(policy = train_var['policy'][0], df = df_train_btc, dir = 'no_fee/basic50', 
  #                       env_params = env_params_basic, total_timesteps = 500000, n_trials=10)
  
  # # Models with fees
  # env_params_basic_fee = {
  #   "initial_position": env_var['initial_position'][0],
  #   "windows": env_var['windows'][1],
  #   "trading_fees": env_var['trading_fees'][1],
  #   "reward_function": env_var['reward_function']
  #   }

  # run_all_optimizations(policy = train_var['policy'][0], df = df_train_btc, dir = 'with_fee/basic50', 
  #                       env_params = env_params_basic_fee, total_timesteps = 500000, n_trials=10)
  


  #--------------------------- OPTIMAL MODELS WITHOUT FEE ------------------------------
  # PPO - without trading fee, window 50, optimal hyperparameters
  # train(model_class = PPO,
  #       algorithm_name = "PPO",
  #       policy = train_var['policy'][0],
  #       df = df_train_btc, 
  #       dir = 'basic/no_fee/PPO/PPO50_opt',
  #       initial_position = env_var['initial_position'][0], 
  #       windows = env_var['windows'][1], 
  #       trading_fees = env_var['trading_fees'][0], 
  #       reward_function = env_var['reward_function'], 
  #       timesteps = train_var['timesteps'],
  #       params_path = "models/params/no_fee/basic50")

  # test_model(df = df_test_btc,
  #           model = PPO,
  #           dir = 'basic/no_fee/PPO/PPO50_opt', 
  #           initial_position = env_var['initial_position'][1], 
  #           windows = env_var['windows'][1],
  #           trading_fees = env_var['trading_fees'][0], 
  #           reward_function = env_var['reward_function'])
  
  # # A2C - without trading fee, window 50, default hyperparameters
  # train(model_class = A2C,
  #       algorithm_name = "A2C",
  #       policy = train_var['policy'][0],
  #       df = df_train_btc, 
  #       dir = 'basic/no_fee/A2C/A2C50_opt',
  #       initial_position = env_var['initial_position'][0], 
  #       windows = env_var['windows'][1], 
  #       trading_fees = env_var['trading_fees'][0], 
  #       reward_function = env_var['reward_function'], 
  #       timesteps = train_var['timesteps'],
  #       params_path = "models/params/no_fee/basic50")

  # test_model(df = df_test_btc,
  #           model = A2C,
  #           dir = 'basic/no_fee/A2C/A2C50_opt', 
  #           initial_position = env_var['initial_position'][1], 
  #           windows = env_var['windows'][1],
  #           trading_fees = env_var['trading_fees'][0], 
  #           reward_function = env_var['reward_function'])
  
  # # DQN - without trading fee, window 50, default hyperparameters
  # train(model_class = DQN,
  #       algorithm_name = "DQN",
  #       policy = train_var['policy'][0],
  #       df = df_train_btc, 
  #       dir = 'basic/no_fee/DQN/DQN50_opt',
  #       initial_position = env_var['initial_position'][0], 
  #       windows = env_var['windows'][1], 
  #       trading_fees = env_var['trading_fees'][0], 
  #       reward_function = env_var['reward_function'], 
  #       timesteps = train_var['timesteps'],
  #       params_path = "models/params/no_fee/basic50")

  # test_model(df = df_test_btc,
  #           model = DQN,
  #           dir = 'basic/no_fee/DQN/DQN50_opt', 
  #           initial_position = env_var['initial_position'][1], 
  #           windows = env_var['windows'][1],
  #           trading_fees = env_var['trading_fees'][0], 
  #           reward_function = env_var['reward_function'])
  

# #--------------------------- MODELS WITH FEE ------------------------------
# # PPO - with trading fee, window 50, default hyperparameters
#   train(model_class = PPO,
#         algorithm_name = "PPO",
#         policy = train_var['policy'][0],
#         df = df_train_btc, 
#         dir = 'basic/with_fee/PPO/PPO50_opt',
#         initial_position = env_var['initial_position'][0], 
#         windows = env_var['windows'][1], 
#         trading_fees = env_var['trading_fees'][1], 
#         reward_function = env_var['reward_function'], 
#         timesteps = train_var['timesteps'],
#         params_path = "models/params/with_fee/basic50")

#   test_model(df = df_test_btc,
#             model = PPO,
#             dir = 'basic/with_fee/PPO/PPO50_opt', 
#             initial_position = env_var['initial_position'][1], 
#             windows = env_var['windows'][1],
#             trading_fees = env_var['trading_fees'][1], 
#             reward_function = env_var['reward_function'])
  
#   # A2C - without trading fee, window 50, default hyperparameters
#   train(model_class = A2C,
#         algorithm_name = "A2C",
#         policy = train_var['policy'][0],
#         df = df_train_btc, 
#         dir = 'basic/with_fee/A2C/A2C50_opt',
#         initial_position = env_var['initial_position'][0], 
#         windows = env_var['windows'][1], 
#         trading_fees = env_var['trading_fees'][1], 
#         reward_function = env_var['reward_function'], 
#         timesteps = train_var['timesteps'],
#         params_path = "models/params/with_fee/basic50")

#   test_model(df = df_test_btc,
#             model = A2C,
#             dir = 'basic/with_fee/A2C/A2C50_opt', 
#             initial_position = env_var['initial_position'][1], 
#             windows = env_var['windows'][1],
#             trading_fees = env_var['trading_fees'][1], 
#             reward_function = env_var['reward_function'])
  
#   # DQN - without trading fee, window 50, default hyperparameters
#   train(model_class = DQN,
#         algorithm_name = "DQN",
#         policy = train_var['policy'][0],
#         df = df_train_btc, 
#         dir = 'basic/with_fee/DQN/DQN50_opt',
#         initial_position = env_var['initial_position'][0], 
#         windows = env_var['windows'][1], 
#         trading_fees = env_var['trading_fees'][1], 
#         reward_function = env_var['reward_function'], 
#         timesteps = train_var['timesteps'],
#         params_path = "models/params/with_fee/basic50")

#   test_model(df = df_test_btc,
#             model = DQN,
#             dir = 'basic/with_fee/DQN/DQN50_opt', 
#             initial_position = env_var['initial_position'][1], 
#             windows = env_var['windows'][1],
#             trading_fees = env_var['trading_fees'][1], 
#             reward_function = env_var['reward_function'])

