import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, stock_data, vix_data, transaction_cost_percent=0.005):
        super(StockTradingEnv, self).__init__()
        
        # Remove any empty DataFrames
        self.stock_data = {ticker: df for ticker, df in stock_data.items()} # this includes VIX, remove VIX
        self.vix_data = vix_data['Close'].to_numpy()
        self.tickers = list(self.stock_data.keys())
        

        # Calculate the size of one stock's data
        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns) # number of features for each stock (i.e price and TAs)
        
        # Define action and observation space
        """
        This line defines a continuous action space where:

        Each element represents an action on a specific ticker.
        Each action is constrained between -1 (sell) and 1 (buy).
        The action space size matches the number of tickers (len(self.tickers)), 
        and each action can independently take a continuous value within the defined range.
        """
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32)
        
        # Observation space: price data for each stock + balance + shares held + net worth + max net worth + current step
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        # values in the observations space can take on values in the continuous range (-inf, inf). Each observation is a long vector containig all the values in the space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        
        # Initialize account balance
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}

        
        # Set the current step
        self.current_step = 0
        
        # Calculate the minimum length of data across all stocks
        self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)
        # max steps is basically the length of our dataset - each day is a step

        # Transaction cost
        self.transaction_cost_percent = transaction_cost_percent
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment back to the start
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        return self._next_observation(), {}
    
    def _next_observation(self):
        # initialize the frame
        frame = np.zeros(self.obs_shape) # empty observation vector
        
        # Add stock data for each ticker
        idx = 0
        # Loop through each ticker
        for ticker in self.tickers:
            # Get the DataFrame for the current ticker
            df = self.stock_data[ticker]
            # If the current step is less than the length of the DataFrame, add the price data for the current step
            if self.current_step < len(df):
                frame[idx:idx+self.n_features] = df.iloc[self.current_step].values
            # Otherwise, add the last price data available
            elif len(df) > 0:
                frame[idx:idx+self.n_features] = df.iloc[-1].values
            # Move the index to the next ticker
            idx += self.n_features
        
        # This look [features for stock1, features for stock2]

        # Add balance, shares held, net worth, max net worth, and current step
        frame[-4-len(self.tickers)] = self.balance # Balance
        frame[-3-len(self.tickers):-3] = [self.shares_held[ticker] for ticker in self.tickers] # Shares held
        frame[-3] = self.net_worth # Net worth
        frame[-2] = self.max_net_worth # Max net worth
        frame[-1] = self.current_step # Current step

        # Adds these observations to the end of the observation vector
        
        return frame
    
    def step(self, actions):
        # update the current step
        self.current_step += 1
        
        # check if we have reached the maximum number of steps
        if self.current_step > self.max_steps:
            return self._next_observation(), 0, True, False, {}
        
        current_prices = {}
        # Loop through each ticker and perform the action
        # if self.vix_data[self.current_step] > 50:
        #     for i, ticker in enumerate(self.tickers):
        #         current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step]['Close']
        #         shares_to_sell = int(self.shares_held[ticker])
        #         sale = shares_to_sell * current_prices[ticker]
        #         # Transaction cost
        #         transaction_cost = sale * self.transaction_cost_percent
        #         # Update the balance and shares held
        #         self.balance += (sale - transaction_cost)
        #         # Update the total shares sold
        #         self.shares_held[ticker] -= shares_to_sell
        #         # Update the shares sold
        #         self.total_shares_sold[ticker] += shares_to_sell
        #         # Update the total sales value
        #         self.total_sales_value[ticker] += sale
        # else:
        for i, ticker in enumerate(self.tickers):
            # Get the current price of the stock
            current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step]['Close']
            # get the action for the current ticker
            action = actions[i]
            # if self.vix_data[self.current_step] < 50:
            if action > 0:  # Buy
                # Calculate the number of shares to buy
                # shares to buy is proportion of balance / current_price
                # spend some portion of your balance on buying shares, action is the portion of your balance to spend
                shares_to_buy = int(self.balance * action / current_prices[ticker])
                # Calculate the cost of the shares
                cost = shares_to_buy * current_prices[ticker]
                # Transaction cost
                transaction_cost = cost * self.transaction_cost_percent
                # Update the balance and shares held
                self.balance -= (cost + transaction_cost)
                # Update the total shares sold
                self.shares_held[ticker] += shares_to_buy

            elif action < 0:  # Sell
                # Calculate the number of shares to sell
                # sell x percentage of your shares for this stock
                shares_to_sell = int(self.shares_held[ticker] * abs(action))
                # Calculate the sale value
                sale = shares_to_sell * current_prices[ticker]
                # Transaction cost
                transaction_cost = sale * self.transaction_cost_percent
                # Update the balance and shares held
                self.balance += (sale - transaction_cost)
                # Update the total shares sold
                self.shares_held[ticker] -= shares_to_sell
                # Update the shares sold
                self.total_shares_sold[ticker] += shares_to_sell
                # Update the total sales value
                self.total_sales_value[ticker] += sale
        
        # Calculate(updating) the net worth
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + sum(self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers)
        # Update the max net worth
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        # Calculate the reward
        # the reward is just net worth so far
        #reward = self.net_worth - self.initial_balance
        reward = self.net_worth - self.prev_net_worth # positive is good

        if self.vix_data[self.current_step] > 30:
            reward -= sum(self.shares_held.values())

        # Check if the episode is done
        done = self.net_worth <= 0 or self.current_step >= self.max_steps
        
        obs = self._next_observation()
        return obs, reward, done, False, {}
    
    def render(self, mode='human'):
        # Print the current step, balance, shares held, net worth, and profit
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        for ticker in self.tickers:
            print(f'{ticker} Shares held: {self.shares_held[ticker]}')
        print(f'Net worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')

    def close(self):
        pass

    def update_stock_data(self, new_stock_data, transaction_cost_percent=None):
        """
        Update the environment with new stock data.

        Parameters:
        new_stock_data (dict): Dictionary containing new stock data,
                            with keys as stock tickers and values as DataFrames.
        """
        # Remove empty DataFrames
        self.stock_data = {ticker: df for ticker, df in new_stock_data.items() if not df.empty}
        self.tickers = list(self.stock_data.keys())

        if not self.tickers:
            raise ValueError("All new stock data are empty")

        # Update the number of features if needed
        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)

        # Update observation space
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

        # Update maximum steps
        self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)

        # Update transaction cost if provided
        if transaction_cost_percent is not None:
            self.transaction_cost_percent = transaction_cost_percent

        # Reset the environment
        self.reset()

        print(f"The environment has been updated with {len(self.tickers)} new stocks.")