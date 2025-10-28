import os
import pandas as pd
import json

# --- Configuration ---
folder_path = "ticker_data"
ATR_PERIOD = 14
SMA_PERIOD = 150 
SMA_SLOPE_PERIOD = 10 
RISE_PERIOD = 5 
columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
true_range_dict = {}

# DataFrame Column Names (Internal)
SMA_COL_NAME = 'SMA_150_ADJ_CLOSE' 
SMA_SLOPE_COL_NAME = 'AVG_SLOPE' 
RISE_COL_NAME = '5_DAY_RISE_PCT' 
ABOVE_AVG_COL_NAME = 'ABOVE_AVG_PCT' 

# Final JSON Key Names
ATR_JSON_KEY = 'atr_14' 
SMA_JSON_KEY = 'sma150_adj_close'
SLOPE_JSON_KEY = '10_day_avg_slope' 
RISE_JSON_KEY = '5_day_rise' 
ABOVE_AVG_JSON_KEY = 'adj_price_above_avg_pct'
INVESTABLE_JSON_KEY = 'investable' # New JSON key

# --- Rule Cleaning Utility ---
def clean_rule_value(value_str):
    """Converts a rule string (e.g., '-5.0%', '$14.00') into a float."""
    if isinstance(value_str, (int, float)):
        return float(value_str)
        
    value_str = str(value_str).strip()
    
    # Handle percentage: remove '%' and convert to float
    if value_str.endswith('%'):
        return float(value_str.strip('%'))
    
    # Handle currency: remove '$' and convert to float
    if value_str.startswith('$'):
        return float(value_str.strip('$'))
    
    # Attempt simple float conversion
    try:
        return float(value_str)
    except ValueError:
        return None # Return None if conversion fails

# Load rules.json
try:
    with open('rules.json', 'r') as f:
        rules_json_data = f.read()
        rules = json.loads(rules_json_data)
        underlying_rules = rules["underlying_stock"]
        
        # Load and clean the five key values into comparable variables
        min_5_day_rise_pct_rule = clean_rule_value(underlying_rules["min_5_day_rise_pct"])
        min_above_avg_pct_rule = clean_rule_value(underlying_rules["min_above_avg_pct"])
        max_above_avg_pct_rule = clean_rule_value(underlying_rules["max_above_avg_pct"])
        min_avg_up_slope_pct_rule = clean_rule_value(underlying_rules["min_avg_up_slope_pct"])
        min_stock_price_rule = clean_rule_value(underlying_rules["min_stock_price"])
        
except Exception as e:
    print(f"⚠️ Error loading or cleaning 'rules.json': {e}. Setting rules to conservative defaults (0.0).")
    min_5_day_rise_pct_rule = 0.0
    min_above_avg_pct_rule = 0.0
    max_above_avg_pct_rule = 100.0 # Effectively no upper limit if rule fails
    min_avg_up_slope_pct_rule = 0.0
    min_stock_price_rule = 0.0

# --- Step 1: Calculate True Range and Prepare Data for ATR/SMA ---
for filename in os.listdir(folder_path):
    if not filename.endswith(".csv"):
        continue

    ticker = filename.replace(".csv", "")
    file_path = os.path.join(folder_path, filename)

    # Read CSV and ensure numeric columns
    df = pd.read_csv(file_path, skiprows=3, header=None, names=columns)
    
    # Convert all necessary columns to numeric types, coercing errors to NaN
    numeric_cols = ['Adj Close', 'High', 'Low', 'Close']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where required values (High, Low, Close, Adj Close) are NaN
    df.dropna(subset=numeric_cols, inplace=True)

    # --- True Range (TR) Calculation (Same as before) ---
    df['Prev Close'] = df['Close'].shift(1) 
    df['TR'] = df[['High', 'Low', 'Close']].apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - df.loc[row.name, 'Prev Close']) if row.name > 0 else 0,
            abs(row['Low'] - df.loc[row.name, 'Prev Close']) if row.name > 0 else 0
        ), 
        axis=1
    )
    df.loc[df.index[0], 'TR'] = df.loc[df.index[0], 'High'] - df.loc[df.index[0], 'Low']
    df['TR_PCT'] = (df['TR'] / df['Close']) * 100

    # --- Step 2: Calculate Moving Averages (Rolling Mean) (Same as before) ---
    df['ATR_14'] = df['TR_PCT'].rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    df[SMA_COL_NAME] = df['Adj Close'].rolling(window=SMA_PERIOD, min_periods=SMA_PERIOD).mean()

    # --- Step 3: Calculate Slope, 5-Day Rise, and Above-Average Percentage (Same as before) ---
    df['SMA_PREV'] = df[SMA_COL_NAME].shift(SMA_SLOPE_PERIOD)
    df[SMA_SLOPE_COL_NAME] = ((df[SMA_COL_NAME] / df['SMA_PREV']) - 1) * 100
    df['ADJ_CLOSE_PREV_5'] = df['Adj Close'].shift(RISE_PERIOD)
    df[RISE_COL_NAME] = ((df['Adj Close'] / df['ADJ_CLOSE_PREV_5']) - 1) * 100
    df[ABOVE_AVG_COL_NAME] = ((df['Adj Close'] / df[SMA_COL_NAME]) - 1) * 100
    
    # --- Step 4: Add 'investable' boolean to DataFrame ---
    # Convert rules to Booleans: All four conditions must be True.
    # We use boolean masks, which will be False wherever NaN prevents a comparison.
    
    # 1. 5-day Rise condition
    cond_rise = (df[RISE_COL_NAME] > min_5_day_rise_pct_rule)
    
    # 2. Above/Below SMA conditions
    cond_above_avg = (df[ABOVE_AVG_COL_NAME] >= min_above_avg_pct_rule) & \
                     (df[ABOVE_AVG_COL_NAME] <= max_above_avg_pct_rule)
                     
    # 3. SMA Slope condition
    cond_slope = (df[SMA_SLOPE_COL_NAME] > min_avg_up_slope_pct_rule)
    
    # 4. Minimum Price condition
    cond_price = (df['Adj Close'] > min_stock_price_rule)
    
    # Combine all conditions
    df[INVESTABLE_JSON_KEY] = cond_rise & cond_above_avg & cond_slope & cond_price
    
    # --- Step 5: Convert DataFrame back to the required dictionary structure ---
    tr_values = {}
    
    for index, row in df.iterrows():
        date = row['Date']
        prev_close_val = row['Prev Close']

        # Base dictionary with all required price and TR data
        tr_values[date] = {
            "true_range": round(row['TR'], 4),
            "true_range_pct": f"{round(row['TR_PCT'], 2)}%",
            "high": round(row['High'], 4),
            "low": round(row['Low'], 4),
            "close": round(row['Close'], 4),
            "prev_close": round(prev_close_val, 4) if not pd.isna(prev_close_val) else None,
            "adj_close": round(row['Adj Close'], 4)
        }
        
        # Add technical metrics
        if not pd.isna(row['ATR_14']):
            tr_values[date][ATR_JSON_KEY] = f"{round(row['ATR_14'], 3):.3f}%"
        if not pd.isna(row[SMA_COL_NAME]):
            tr_values[date][SMA_JSON_KEY] = round(row[SMA_COL_NAME], 4)
        if not pd.isna(row[ABOVE_AVG_COL_NAME]):
            tr_values[date][ABOVE_AVG_JSON_KEY] = f"{round(row[ABOVE_AVG_COL_NAME], 3):.3f}%"
        if not pd.isna(row[SMA_SLOPE_COL_NAME]):
            tr_values[date][SLOPE_JSON_KEY] = f"{round(row[SMA_SLOPE_COL_NAME], 3):.3f}%"
        if not pd.isna(row[RISE_COL_NAME]):
            tr_values[date][RISE_JSON_KEY] = f"{round(row[RISE_COL_NAME], 3):.3f}%"

        # Add 'investable' boolean (The result is already a simple True/False)
        # Note: We use .item() to extract the single boolean value from the Pandas Series/object
        # The result will be None if any underlying calculation was NaN.
        tr_values[date][INVESTABLE_JSON_KEY] = bool(row[INVESTABLE_JSON_KEY]) if pd.notna(row[INVESTABLE_JSON_KEY]) else False


    true_range_dict[ticker] = tr_values
    print(f"✅ Calculated all metrics and '{INVESTABLE_JSON_KEY}' flag for {ticker} ({len(tr_values)} days)")

# --- Step 6: Save the final dictionary to JSON once ---
with open("true_range.json", "w") as f:
    json.dump(true_range_dict, f, indent=4)

print("✅ Saved true_range.json")