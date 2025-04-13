
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

