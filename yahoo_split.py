import yfinance as yf
import pandas as pd
from datetime import datetime
import json
import os

# Define date range
start_date = "2006-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
overwrite = True  # Flag to control overwriting of existing files

def process_ticker(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)

    # Get historical price data
    hist = ticker.history(start=start_date, end=end_date)  # download history

    # Get dividends and splits
    dividends = ticker.dividends
    splits = ticker.splits

    # Create a DataFrame with dividends and splits aligned with dates
    hist['Dividends'] = dividends
    hist['Stock Splits'] = splits

    # Fill missing dividends/splits with 0
    hist['Dividends'] = hist['Dividends'].fillna(0)
    hist['Stock Splits'] = hist['Stock Splits'].fillna(0)

    # Set Adj_Close to the original Close value before any splits
    hist['Adj_Close'] = hist['Close']
    split_dates = hist[hist['Stock Splits'] > 0].index
    for split_date in split_dates:
        split_ratio = hist.loc[split_date, 'Stock Splits']
        hist.loc[:split_date - pd.Timedelta(days=1), ['Close', 'High', 'Low', 'Open']] *= split_ratio

    # Update Split_Ratio to show split ratio only on split dates
    hist['Split_Ratio'] = hist['Stock Splits']

    # Ensure the directory exists
    os.makedirs("ticker_data", exist_ok=True)

    # Define file path
    file_path = f"ticker_data/{ticker_symbol}.csv"

    # Check overwrite flag
    if not overwrite and os.path.exists(file_path):
        print(f"Skipped {file_path} (file exists and overwrite is False)")
        return

    # Ensure the index is reset to include 'Date' as a column
    hist.reset_index(inplace=True)

    # Ensure 'Date' column is in datetime format
    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')

    # Save to CSV
    output_columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Dividends', 'Split_Ratio']
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')  # Format date as yyyy-mm-dd
    hist.to_csv(file_path, columns=output_columns, index=False)

    # Check if the file has less than 150 rows and delete it if true
    if len(hist) < 150:
        os.remove(file_path)
        print(f"Deleted {file_path} (less than 150 rows)")
    else:
        print(f"Saved   {file_path}")

# Load tickers
with open('tickers.json', 'r') as f:
    tickers = json.load(f)

# Process all tickers
for ticker in tickers:
    # if ticker == "TQQQ":
    process_ticker(ticker)