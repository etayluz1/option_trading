import os
import pandas as pd
import json

# --- Configuration ---
folder_path = "ticker_data"
ATR_PERIOD = 14
SMA_PERIOD = 150 
SMA_SLOPE_PERIOD = 10 
RISE_PERIOD = 5 
#  Date,Adj_Close,Close,High,Low,Open,Dividends,Split_Ratio:
columns = ['Date', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Dividends', 'Split_Ratio']
stock_history_dict = {}

# DataFrame Column Names (Internal)
SMA_COL_NAME = 'SMA_150_ADJ_CLOSE' 
SMA_SLOPE_COL_NAME = 'AVG_SLOPE' 
RISE_COL_NAME = 'tead_PCT' 
ABOVE_AVG_COL_NAME = 'ABOVE_AVG_PCT' 
BELOW_AVG_COL_NAME = 'BELOW_AVG_PCT' 
EXIT_COL_NAME = 'SHOULD_EXIT' # New internal column name

# Final JSON Key Names
ATR_JSON_KEY = 'atr_14' 
SMA_JSON_KEY = 'sma150_adj_close'
SLOPE_JSON_KEY = '10_day_avg_slope' 
RISE_JSON_KEY = '5_day_rise' 
ABOVE_AVG_JSON_KEY = 'adj_price_above_avg_pct'
BELOW_AVG_JSON_KEY = 'adj_close_below_avg_pct' 
INVESTABLE_JSON_KEY = 'investable' 
SHOULD_EXIT_JSON_KEY = 'should_exit' # New JSON key

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
        return None 

# Load rules.json
try:
    with open('rules.json', 'r') as f:
        rules_json_data = f.read()
        rules = json.loads(rules_json_data)
        underlying_rules = rules["underlying_stock"]
        exit_put_position = rules["exit_put_position"]
        
        # Load and clean rules
        min_5_day_rise_pct_rule = clean_rule_value(underlying_rules["min_5_day_rise_pct"])
        min_above_avg_pct_rule = clean_rule_value(underlying_rules["min_above_avg_pct"])
        max_above_avg_pct_rule = clean_rule_value(underlying_rules["max_above_avg_pct"])
        min_avg_up_slope_pct_rule = clean_rule_value(underlying_rules["min_avg_up_slope_pct"])
        min_stock_price_rule = clean_rule_value(underlying_rules["min_stock_price"])
        stock_max_below_avg_rule = clean_rule_value(exit_put_position["stock_max_below_avg"])
        
except Exception as e:
    print(f"⚠️ Error loading or cleaning 'rules.json': {e}. Setting rules to conservative defaults.")
    min_5_day_rise_pct_rule = 0.0
    min_above_avg_pct_rule = 0.0
    max_above_avg_pct_rule = 100.0 
    min_avg_up_slope_pct_rule = 0.0
    min_stock_price_rule = 0.0
    stock_max_below_avg_rule = 0.0

# --- Step 1: Calculate True Range and Prepare Data for ATR/SMA ---
for filename in os.listdir(folder_path):
    if not filename.endswith(".csv"):
        continue

    ticker = filename.replace(".csv", "")
    file_path = os.path.join(folder_path, filename)

    # Read CSV and ensure numeric columns
    df = pd.read_csv(file_path, skiprows=3, header=None, names=columns, dtype={"Date": str})
    
    # Convert all necessary columns to numeric types, coercing errors to NaN
    numeric_cols = ['Adj_Close', 'High', 'Low', 'Close']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where required values (High, Low, Close, Adj Close) are NaN
    df.dropna(subset=numeric_cols, inplace=True)

    # Ensure the index is numeric
    df.reset_index(drop=True, inplace=True)

    # Ensure the Date column is explicitly converted to a string after reading
    df['Date'] = df['Date'].astype(str)

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
    df[SMA_COL_NAME] = df['Adj_Close'].rolling(window=SMA_PERIOD, min_periods=SMA_PERIOD).mean()

    # --- Step 3: Calculate Slope, 5-Day Rise, and Above/Below-Average Percentage ---
    
    # A. Average Slope Calculation
    df['SMA_PREV'] = df[SMA_COL_NAME].shift(SMA_SLOPE_PERIOD)
    df[SMA_SLOPE_COL_NAME] = ((df[SMA_COL_NAME] / df['SMA_PREV']) - 1) * 100
    
    # B. 5-Day Rise Calculation
    df['ADJ_CLOSE_PREV_5'] = df['Adj_Close'].shift(RISE_PERIOD)
    df[RISE_COL_NAME] = ((df['Adj_Close'] / df['ADJ_CLOSE_PREV_5']) - 1) * 100
    
    # C. Above-Average Percentage Calculation
    df[ABOVE_AVG_COL_NAME] = ((df['Adj_Close'] / df[SMA_COL_NAME]) - 1) * 100

    # D. Below-Average Percentage Calculation (Negation)
    df[BELOW_AVG_COL_NAME] = -1 * df[ABOVE_AVG_COL_NAME]
    
    # --- Step 4: Add 'investable' and 'should_exit' booleans to DataFrame ---
    
    # INVESTABLE LOGIC (Entry Screen)
    cond_rise = (df[RISE_COL_NAME] > min_5_day_rise_pct_rule)
    cond_above_avg = (df[ABOVE_AVG_COL_NAME] >= min_above_avg_pct_rule) & \
                     (df[ABOVE_AVG_COL_NAME] <= max_above_avg_pct_rule)
    cond_slope = (df[SMA_SLOPE_COL_NAME] > min_avg_up_slope_pct_rule)
    cond_price = (df['Adj_Close'] > min_stock_price_rule)
    df[INVESTABLE_JSON_KEY] = cond_rise & cond_above_avg & cond_slope & cond_price
    
    # SHOULD_EXIT LOGIC (Exit Screen - NEW)
    # True if adj_close_below_avg_pct is LARGER than the stock_max_below_avg_rule
    # Note: Both values are in absolute percentage terms (e.g., 5.0).
    df[EXIT_COL_NAME] = (df[BELOW_AVG_COL_NAME] > stock_max_below_avg_rule)

# ... (Steps 1 to 4 remain the same) ...
    # --- Step 5: Convert DataFrame back to the required dictionary structure ---
    tr_values = {}
    
    for index, row in df.iterrows():
        # Ensure the date is explicitly converted to a string for JSON output
        date_str = str(row['Date'])



        # Enforce the exact key order as specified by the user
        from collections import OrderedDict
        # Always output keys in the exact order, using None for missing values
        from collections import OrderedDict
        def safe_val(val, fmt=None, dec=None):
            if pd.isna(val):
                return None
            if fmt == 'pct':
                return f"{round(val, 3):.3f}%"
            if dec is not None:
                return round(val, dec)
            return val

        key_order = [
            ('adj_close', safe_val(row['Adj_Close'], dec=5)),
            ('close', safe_val(row['Close'], dec=5)),
            ('5_day_rise', safe_val(row[RISE_COL_NAME], fmt='pct')),
            ('10_day_avg_slope', safe_val(row[SMA_SLOPE_COL_NAME], fmt='pct')),
            ('adj_price_above_avg_pct', safe_val(row[ABOVE_AVG_COL_NAME], fmt='pct')),
            ('sma150_adj_close', safe_val(row[SMA_COL_NAME], dec=4)),
            ('Split', safe_val(row['Split_Ratio']) if 'Split_Ratio' in df.columns else None)
        ]
        date_metrics = OrderedDict((k, v) for k, v in key_order if v is not None)
        tr_values[date_str] = date_metrics

    stock_history_dict[ticker] = tr_values
    print(f"✅ Calculated all metrics for {ticker} ({len(tr_values)} days)")

# --- Step 6: Save the final dictionary to JSON once ---
# The 'sort_keys=True' argument in json.dump will sort the top-level keys

# Use sort_keys=False to preserve OrderedDict key order for date metrics
with open("stock_history.json", "w") as f:
    json.dump(stock_history_dict, f, indent=4, sort_keys=False)

print("✅ Saved stock_history.json with all keys in alphabetical order.")