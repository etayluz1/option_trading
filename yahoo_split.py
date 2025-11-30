import yfinance as yf
import pandas as pd
from datetime import datetime

# Replace 'HDLC' with your ticker
ticker_symbol = "TQQQ"
ticker = yf.Ticker(ticker_symbol)

# Define date range
start_date = "2006-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# Get historical price data
hist = ticker.history(start=start_date, end=end_date)  # max history available

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

# Save to CSV
output_columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Dividends', 'Split_Ratio']
hist.reset_index(inplace=True)
hist.to_csv(f"{ticker_symbol}_full_history.csv", columns=output_columns, index=False)

print(f"Saved combined data with adjusted prices to {ticker_symbol}_full_history.csv")