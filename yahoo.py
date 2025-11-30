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
    hist_original = ticker.history(start=start_date, end=end_date)  # download history    
    os.makedirs("ticker_data", exist_ok=True)
    hist_orig_csv = hist_original.copy().reset_index()
    hist_orig_csv['Date'] = pd.to_datetime(hist_orig_csv['Date'], errors='coerce').dt.strftime('%Y-%m-%d')    
    save_original = False  # Flag to control saving of original data
    if save_original:
        # Save the original data to a separate CSV       
        file_path_original = f"ticker_data/{ticker_symbol}_original.csv" 
        hist_orig_csv.to_csv(file_path_original, index=False)

    # Work on a copy for all adjustments
    hist = hist_original.copy()
    # Ensure index is DatetimeIndex for all adjustments
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index)

    # Get dividends and splits
    dividends = ticker.dividends
    splits = ticker.splits

    # Create a DataFrame with dividends and splits aligned with dates
    hist['Dividends'] = dividends
    hist['Stock Splits'] = splits

    # Fill missing dividends/splits with 0
    hist['Dividends'] = hist['Dividends'].fillna(0)
    hist['Stock Splits'] = hist['Stock Splits'].fillna(0)


    # Set Adj_Close to the original Close value before any splits/dividends
    hist['Adj_Close'] = hist['Close']

    # Adjust for splits (multiplicative)
    split_dates = hist[hist['Stock Splits'] > 0].index
    for split_date in split_dates:
        split_ratio = hist.loc[split_date, 'Stock Splits']
        hist.loc[:split_date - pd.Timedelta(days=1), ['Close', 'High', 'Low', 'Open']] *= split_ratio

    # Adjust for dividends (multiplicative adjustment factor)
    dividend_dates = hist[hist['Dividends'] > 0].index
    for div_date in dividend_dates:
        dividend_amt = hist.loc[div_date, 'Dividends']
        prev_date = div_date - pd.Timedelta(days=1)
        if prev_date in hist.index:
            close_before = hist.loc[prev_date, 'Close']
            if close_before > 0:
                adj_factor = (close_before - dividend_amt) / close_before
                hist.loc[:prev_date, ['Close', 'High', 'Low', 'Open']] /= adj_factor

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

    # Only after all adjustments, reset index and convert 'Date' to string for saving
    # Save the original dividends before any adjustments
    orig_dividends = hist['Dividends'].copy()
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    # Merge the original dividends back in for output
    hist['Dividends_orig'] = orig_dividends.values
    hist['Dividends'] = hist['Dividends_orig'].combine_first(hist['Dividends'])
    hist.drop(columns=['Dividends_orig'], inplace=True)

    # Save to CSV
    output_columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Dividends', 'Split_Ratio']
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
    process_ticker(ticker)