import yfinance as yf
import pandas as pd

# script to install data

tickers = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DOW',
    'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'TRV', 'UNH', 'RTX', 'VZ', 'V', 'WBA', 'WMT', 'XOM'
]

current_tickers = [
    'MMM', 'AMZN', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO',
    'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'DIS', '^VIX', '^DJI'
]

def get_data(tickers):
    """
    Make a dictionary where the key is a ticker and the value 
    is the daily prices in a datframe
    """
    stock_data = {}
    for ticker in tickers:
        # market recovered June 2009
        df = yf.download(ticker, start="2009-01-01", end="2024-11-01") # was start="2009-01-01", end="2023-05-08"
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        df.index = df.index.date
        # Name the index
        df.index.name = 'Date'
        stock_data[ticker] = df
    return stock_data

stock_data = get_data(current_tickers)

# Explicit column names to avoid confusion
column_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Save to CSV with explicit headers
for ticker, df in stock_data.items():
    # Reset the column names just in case
    df.columns = column_names
    # Ensure the 'Date' is used as the index in the CSV
    df.to_csv(f'data_current/{ticker}.csv', index=True)



# Extract the 'Close' prices
dow_close_prices = stock_data['^DJI']['Close']

# Initial investment amount
initial_investment = 10000

shares_purchased = initial_investment / 26076 # price at 2018-01-30, which is when we would start

# Simulate portfolio value over time
portfolio_value = dow_close_prices * shares_purchased

# Combine dates and portfolio values into a DataFrame
dji_portfolio_df = pd.DataFrame({
    'Date': portfolio_value.index,
    'Portfolio Value': portfolio_value.values
})

# Save the DataFrame to a CSV file
dji_portfolio_df.to_csv("portfolio_value.csv", index=False)