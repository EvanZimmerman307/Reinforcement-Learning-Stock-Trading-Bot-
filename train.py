import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from trading_env import StockTradingEnv
from agents import SimpleAvgEnsembleAgent, A2CAgent, PPOAgent, A2CAgent, DDPGAgent, TD3Agent, SACAgent, WeightedAvgEnsembleAgent
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


def add_technical_indicators(df):
    df = df.copy()
    # MACD and Signal
    df.loc[:, 'EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df.loc[:, 'EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df.loc[:, 'MACD'] = df['EMA12'] - df['EMA26']
    df.loc[:, 'Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
    
    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df.loc[:, 'CCI'] = (tp - sma_tp) / (0.015 * mean_dev)
    
    # ADX
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df.loc[:, '+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df.loc[:, '-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    df.loc[:, '+DI'] = 100 * (df['+DM'].ewm(span=14, adjust=False).mean() / atr)
    df.loc[:, '-DI'] = 100 * (df['-DM'].ewm(span=14, adjust=False).mean() / atr)
    dx = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df.loc[:, 'ADX'] = dx.ewm(span=14, adjust=False).mean()

    # Drop NaN values
    df.dropna(inplace=True)

    # Keep only the relevant columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'RSI', 'CCI', 'ADX']]

    return df

def create_env_and_train_agents(data, vix_data, timesteps):
    # Create the environment using DummyVecEnv with training data
    env = DummyVecEnv([lambda: StockTradingEnv(data, vix_data)])

    # Train PPO Agent
    ppo_agent = PPOAgent(env=env, total_timesteps=timesteps)

    # Train A2C Agent
    a2c_agent = A2CAgent(env=env, total_timesteps=timesteps)

    # Train DDPG Agent
    ddpg_agent = DDPGAgent(env=env, total_timesteps=timesteps)

    # Train SAC Agent
    sac_agent = SACAgent(env=env, total_timesteps=timesteps)

    # Train TD3 Agent
    td3_agent = TD3Agent(env=env, total_timesteps=timesteps)


    return env, ppo_agent, a2c_agent, ddpg_agent, sac_agent, td3_agent


# Stocks from the Dow 30
tickers = [
    'MMM', 'AMZN', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO',
    'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'DIS'
]

# Get the data from the CSV files
stock_data = {}
for ticker in tickers:
    df = pd.read_csv(f'data_current/{ticker}.csv', index_col='Date', parse_dates=True)
    stock_data[ticker] = df

# the date ranges matter a lot
# vix_data = None
vix_data = pd.read_csv(f'data_current/^VIX.csv', index_col='Date', parse_dates=True)
# split the data into training, validation and test sets
training_data_time_range = ('2009-01-01', '2016-12-31') # 70% #  '2009-06-01', '2020-03-18' 7 years
validation_data_time_range = ('2017-01-01', '2017-12-31') # 15% '2020-03-19', '2022-07-11')
test_data_time_range = ('2018-01-01', '2022-05-08') # 15% '2022-07-12', '2024-11-01' 5 1/2 years


# split the data into training, validation and test sets
training_data = {}
validation_data = {}
test_data = {}

# split the data dictionary into subdictionaries for training, validation, testing
for ticker, df in stock_data.items():
    training_data[ticker] = df.loc[training_data_time_range[0]:training_data_time_range[1]]
    validation_data[ticker] = df.loc[validation_data_time_range[0]:validation_data_time_range[1]]
    test_data[ticker] = df.loc[test_data_time_range[0]:test_data_time_range[1]]
vix_training_data = vix_data.loc[training_data_time_range[0]:training_data_time_range[1]]
vix_validation_data = vix_data.loc[validation_data_time_range[0]:validation_data_time_range[1]]
vix_test_data = vix_data.loc[test_data_time_range[0]:test_data_time_range[1]]
# add technical indicators to the training data for each stock
for ticker, df in training_data.items():
    training_data[ticker] = add_technical_indicators(df)

# add technical indicators to the validation data for each stock
for ticker, df in validation_data.items():
    validation_data[ticker] = add_technical_indicators(df)

# add technical indicators to the test data for each stock
for ticker, df in test_data.items():
    test_data[ticker] = add_technical_indicators(df)

# print shape of training, validation and test data
ticker = 'MMM'
print(f'Training data shape for {ticker}: {training_data[ticker].shape}')
print(f'Validation data shape for {ticker}: {validation_data[ticker].shape}')
print(f'Test data shape for {ticker}: {test_data[ticker].shape}')

print(test_data["MMM"].head())

# Create the environment and train the agents
total_timesteps = 10000 # 10,000 days of training, try 20?
env, ppo_agent, a2c_agent, ddpg_agent, sac_agent, td3_agent = create_env_and_train_agents(training_data, vix_training_data, total_timesteps)



















