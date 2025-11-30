import yfinance as yf
import pandas as pd

# Download OHLCV + splits
tqqq = yf.Ticker("TQQQ")
hist = tqqq.history(period="max")

# Flatten column names if multi-index is present
if isinstance(hist.columns, pd.MultiIndex):
    hist.columns = [col[0] for col in hist.columns]

# Use the correct column name for splits
if 'Stock Splits' in hist.columns:
    hist["Split_Ratio"] = hist['Stock Splits'].fillna(1)
else:
    raise KeyError("Splits column key could not be determined. Check yfinance version.")

# Calculate cumulative split ratio
hist["Cumulative_Split_Ratio"] = hist["Split_Ratio"].cumprod()

# Adjust prices for Open, High, Low, Close
hist["Adj_Open"] = hist["Open"] / hist["Cumulative_Split_Ratio"]
hist["Adj_High"] = hist["High"] / hist["Cumulative_Split_Ratio"]
hist["Adj_Low"] = hist["Low"] / hist["Cumulative_Split_Ratio"]
hist["Adj_Close"] = hist["Close"] / hist["Cumulative_Split_Ratio"]

# Debugging: Print column names and first few rows to verify adjusted columns are present
print("Columns in DataFrame:", hist.columns)
print("First few rows of DataFrame:")
print(hist.head())

# Ensure all columns are saved
output_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Capital Gains", "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close"]
hist.reset_index(inplace=True)
hist.to_csv("TQQQ_full_history.csv", columns=output_columns, index=False)
print("Saved TQQQ_full_history.csv with adjusted prices.")