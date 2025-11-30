import yfinance as yf
import pandas as pd

# Replace 'HDLC' with your ticker
ticker_symbol = "TQQQ"
ticker = yf.Ticker(ticker_symbol)

# Get historical price data
hist = ticker.history(period="max")  # max history available

# Get dividends and splits
dividends = ticker.dividends
splits = ticker.splits

# Create a DataFrame with dividends and splits aligned with dates
hist['Dividends'] = dividends
hist['Stock Splits'] = splits

# Fill missing dividends/splits with 0
hist['Dividends'] = hist['Dividends'].fillna(0)
hist['Stock Splits'] = hist['Stock Splits'].fillna(0)

# Save to CSV
hist.to_csv(f"{ticker_symbol}_full_history.csv")

print(f"Saved combined data to {ticker_symbol}_full_history.csv")