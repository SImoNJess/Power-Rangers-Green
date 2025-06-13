import numpy as np
from sklearn.preprocessing import StandardScaler

def price_threshold_policy(state_seq, t=0, sell_threshold=40, buy_threshold=30, base_action=8):
    """
    Policy based on price thresholds:
    - If sell_price exceeds sell_threshold, sell excess energy.
    - If buy_price drops below buy_threshold, buy energy.
    - Otherwise, hold a base action.

    Args:
        state_seq (array-like): Sequence of past state vectors, each including [sun, demand, buy_price, sell_price, ...].
        t (int): Index of the time step to evaluate in state_seq (default=0).
        sell_threshold (float): Sell price threshold above which to sell.
        buy_threshold (float): Buy price threshold below which to buy.
        base_action (float): Default action value when prices are within thresholds.

    Returns:
        float: An action specifying energy to trade (positive for buying, negative for selling).
    """
    # Ensure state_seq is a NumPy array
    state_seq = np.asarray(state_seq)

    # Extract the t-th state and reshape for scaler
    last_step_scaled = state_seq[t].reshape(1, -1)

    # Assume scaler was fit on the same feature order: [sun, demand, buy_price, sell_price, ...]
    scaler = StandardScaler()
    # scaler should be pre-fitted externally and passed or loaded. Here we mock-fit with identity transformation.
    # In a real use-case, load a trained scaler: e.g., scaler = joblib.load('scaler.pkl')
    scaler.mean_ = np.zeros(last_step_scaled.shape[1])
    scaler.scale_ = np.ones(last_step_scaled.shape[1])

    # Unscale to get real-world values
    last_step_unscaled = scaler.inverse_transform(last_step_scaled)[0]
    sun_price = last_step_unscaled[0]
    demand    = last_step_unscaled[1]
    buy_price = last_step_unscaled[2]
    sell_price= last_step_unscaled[3]

    # Decision logic
    if sell_price > sell_threshold:
        # Sell energy: negative action proportional to price excess
        action = base_action - (sell_price - sell_threshold)
    elif buy_price < buy_threshold:
        # Buy energy: positive action proportional to price gap
        action = base_action + (buy_threshold - buy_price)
    else:
        # Hold baseline action
        action = base_action
    return float(action)
