import sys
import subprocess

try:
    import yfinance as yf
except ModuleNotFoundError:
    print("yfinance not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

import json
import os
from datetime import datetime
Overwrite = True # Flag to control overwriting

# Load tickers
with open('tickers.json', 'r') as f:
    tickers = json.load(f)

# Define date range
start_date = "2006-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# Make sure output directory exists
os.makedirs("ticker_data", exist_ok=True)

for ticker in tickers:
    file_path = f"ticker_data/{ticker}.csv"

    # Skip if file already exists and Overwrite is False
    if not Overwrite and os.path.exists(file_path):
        print(f"⏩ Skipping {ticker} — file already exists.")
        continue

    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

    # Handle empty data
    if data.empty:
        print(f"⚠️ No data found for {ticker}. Skipping.")
        continue

    # Save to CSV
    data.to_csv(file_path)
    print(f"✅ Saved {file_path} with {len(data)} rows")
