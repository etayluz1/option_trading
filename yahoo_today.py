import yfinance as yf
import pandas as pd
from datetime import datetime
import orjson
import os
from collections import OrderedDict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import subprocess

# Tradier API configuration
TRADIER_API_KEY = "6IyR6wDsuQ2tzm9mGxzLDQY1GrTF"
TRADIER_BASE_URL = "https://api.tradier.com/v1"

# Start timing
start_time = time.time()

# Define date range
start_date = (datetime.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%Y-%m-%d")

# Configuration
SMA_PERIOD = 150 
SMA_SLOPE_PERIOD = 10 
RISE_PERIOD = 5

# --- Step 1: Generate rules_all.json with extreme values ---
import json
import glob

def generate_rules_all():
    """Generate rules_all.json with extreme (least restrictive) values from all rules files"""
    rules_files = glob.glob("rules*.json")
    # Exclude rules_all.json (output) and rules.json (different project)
    rules_files = [f for f in rules_files if f not in ["rules_all.json", "rules.json"]]
    
    if not rules_files:
        print("⚠️ No rules files found")
        return
    
    all_rules = []
    for rf in rules_files:
        with open(rf, 'r') as f:
            all_rules.append(json.load(f))
    
    def parse_val(v):
        if v is None:
            return None
        if isinstance(v, (int, float, bool)):
            return v
        if isinstance(v, str):
            # Skip date strings and other non-numeric strings
            if '/' in v and v.count('/') == 2:  # Date format like 01/01/2017
                return None
            try:
                return float(v.replace('%', '').replace('$', ''))
            except ValueError:
                return None
        return v
    
    def format_val(v, original):
        if v is None:
            return original  # Return original if value is None
        if isinstance(original, str):
            if '%' in original:
                return f"{v}%"
            if '$' in original:
                return f"${v}"
            if '.' in original or '.' in str(v):
                return str(v)
            try:
                return str(int(v)) if v == int(v) else str(v)
            except (ValueError, TypeError):
                return str(v)
        return v
    
    def get_extreme(key, values, original_vals):
        """Get min for 'min_' keys, max for 'max_' keys"""
        parsed = [parse_val(v) for v in values if parse_val(v) is not None]
        if not parsed:
            return values[0] if values else None
        
        # For min_ rules, take the minimum (least restrictive)
        # For max_ rules, take the maximum (least restrictive)
        if 'min_' in key or key.startswith('min'):
            result = min(parsed)
        elif 'max_' in key or key.startswith('max'):
            result = max(parsed)
        else:
            result = parsed[0]  # Non min/max rules - take first
        
        # Find original format
        for orig in original_vals:
            if orig is not None:
                return format_val(result, orig)
        return result
    
    # Build rules_all from extremes
    rules_all = {}
    base = all_rules[0]
    
    for section, section_data in base.items():
        rules_all[section] = {}
        if isinstance(section_data, dict):
            for key, val in section_data.items():
                values = [r.get(section, {}).get(key) for r in all_rules]
                if any(isinstance(v, bool) for v in values):
                    rules_all[section][key] = val  # Keep booleans as-is
                elif any(isinstance(v, str) and not any(c in v for c in '%$0123456789.-') for v in values if v):
                    rules_all[section][key] = val  # Keep non-numeric strings as-is
                else:
                    rules_all[section][key] = get_extreme(key, values, values)
        else:
            rules_all[section] = section_data
    
    with open("rules_all.json", 'w') as f:
        json.dump(rules_all, f, indent=4)
    
    print(f"[OK] Generated rules_all.json from {len(rules_files)} rules files: {', '.join(rules_files)}")
    return rules_all


def get_tradier_quote(ticker_symbol):
    """Get current quote from Tradier API"""
    # Handle special ticker symbols (Tradier uses forward slash)
    if ticker_symbol in ["BF.B", "BF-B"]:
        ticker_symbol = "BF/B"
    elif ticker_symbol in ["BRK.B", "BRK-B"]:
        ticker_symbol = "BRK/B"
    elif ticker_symbol == "FI":
        ticker_symbol = "FISV"
    
    url = f"{TRADIER_BASE_URL}/markets/quotes?symbols={ticker_symbol}"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "quotes" in data and "quote" in data["quotes"]:
                quote = data["quotes"]["quote"]
                return {
                    "close": quote.get("close") or quote.get("last"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "open": quote.get("open")
                }
    except Exception as e:
        print(f"Tradier error for {ticker_symbol}: {e}")
    return None

def get_option_expirations(symbol="AAPL"):
    """Get all available option expiration dates from Tradier API"""
    url = f"{TRADIER_BASE_URL}/markets/options/expirations?symbol={symbol}"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "expirations" in data and "date" in data["expirations"]:
                # Return list of expiration dates
                return data["expirations"]["date"]
    except Exception as e:
        print(f"Tradier expiration error for {symbol}: {e}")
    return []

def get_put_chain(stock_ticker, expiration_date):
    """
    Get all put options for a stock ticker and expiration date from Tradier API.
    
    Args:
        stock_ticker: Stock symbol (e.g., 'AAPL')
        expiration_date: Expiration date string (e.g., '2025-01-17')
    
    Returns:
        List of put option dictionaries with greeks
    """
    # Handle special ticker symbols (Tradier uses forward slash)
    ticker = stock_ticker
    if stock_ticker in ["BF.B", "BF-B"]:
        ticker = "BF/B"
    elif stock_ticker in ["BRK.B", "BRK-B"]:
        ticker = "BRK/B"
    elif stock_ticker == "FI":
        ticker = "FISV"
    
    url = f"{TRADIER_BASE_URL}/markets/options/chains?symbol={ticker}&expiration={expiration_date}&greeks=true"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    
    puts = []
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "options" in data and data["options"] and "option" in data["options"]:
                for option in data["options"]["option"]:
                    # Only include puts
                    if option.get("option_type") == "put":
                        greeks = option.get("greeks", {}) or {}
                        puts.append({
                            "symbol": option.get("symbol"),
                            "strike": option.get("strike"),
                            "bid": option.get("bid"),
                            "ask": option.get("ask"),
                            "bid_date": option.get("bid_date"),
                            "ask_date": option.get("ask_date"),
                            "last_price": option.get("last"),
                            "volume": option.get("volume"),
                            "open_interest": option.get("open_interest"),
                            "delta": greeks.get("delta"),
                            "gamma": greeks.get("gamma"),
                            "theta": greeks.get("theta"),
                            "vega": greeks.get("vega"),
                            "rho": greeks.get("rho"),
                            "mid_iv": greeks.get("mid_iv"),
                            "expiration_date": option.get("expiration_date")
                        })
    except Exception as e:
        print(f"Tradier put chain error for {stock_ticker}: {e}")
    
    return puts


def process_ticker(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist_original = ticker.history(start=start_date, end=end_date)
    if len(hist_original) < 160:
        print(f"Skipped {ticker_symbol} (insufficient data: {len(hist_original)} days)")
        return None
    hist = hist_original.copy()
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index)
    dividends = ticker.dividends
    splits = ticker.splits
    hist['Dividends'] = dividends
    hist['Stock Splits'] = splits
    hist['Dividends'] = hist['Dividends'].fillna(0)
    hist['Stock Splits'] = hist['Stock Splits'].fillna(0)
    hist['Adj_Close'] = hist['Close']
    split_dates = hist[hist['Stock Splits'] > 0].index
    for split_date in split_dates:
        split_ratio = hist.loc[split_date, 'Stock Splits']
        hist.loc[:split_date - pd.Timedelta(days=1), ['Close', 'High', 'Low', 'Open']] *= split_ratio
    dividend_dates = hist[hist['Dividends'] > 0].index
    for div_date in dividend_dates:
        dividend_amt = hist.loc[div_date, 'Dividends']
        prev_date = div_date - pd.Timedelta(days=1)
        if prev_date in hist.index:
            close_before = hist.loc[prev_date, 'Close']
            if close_before > 0:
                adj_factor = (close_before - dividend_amt) / close_before
                hist.loc[:prev_date, ['Close', 'High', 'Low', 'Open']] /= adj_factor
    hist['Split_Ratio'] = hist['Stock Splits']
    hist['SMA_150_ADJ_CLOSE'] = hist['Adj_Close'].rolling(window=SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    hist['SMA_PREV'] = hist['SMA_150_ADJ_CLOSE'].shift(SMA_SLOPE_PERIOD)
    hist['AVG_SLOPE'] = ((hist['SMA_150_ADJ_CLOSE'] / hist['SMA_PREV']) - 1) * 100
    hist['ADJ_CLOSE_PREV_5'] = hist['Adj_Close'].shift(RISE_PERIOD)
    hist['5_DAY_RISE'] = ((hist['Adj_Close'] / hist['ADJ_CLOSE_PREV_5']) - 1) * 100
    hist['ABOVE_AVG_PCT'] = ((hist['Adj_Close'] / hist['SMA_150_ADJ_CLOSE']) - 1) * 100
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    last_row = hist.iloc[-1]
    yahoo_date = str(last_row['Date'])
    if yahoo_date != today_str:
        tradier_quote = get_tradier_quote(ticker_symbol)
        if tradier_quote and tradier_quote.get("close"):
            new_row = hist.iloc[-1].copy()
            new_row['Date'] = today_str
            new_row['Close'] = tradier_quote['close']
            new_row['Adj_Close'] = tradier_quote['close']
            new_row['High'] = tradier_quote.get('high', tradier_quote['close'])
            new_row['Low'] = tradier_quote.get('low', tradier_quote['close'])
            new_row['Open'] = tradier_quote.get('open', tradier_quote['close'])
            new_row['Dividends'] = 0
            new_row['Stock Splits'] = 0
            new_row['Split_Ratio'] = 0
            prev_5_adj_close = hist.iloc[-RISE_PERIOD]['Adj_Close'] if len(hist) >= RISE_PERIOD else hist.iloc[-1]['Adj_Close']
            last_149_adj_close = hist['Adj_Close'].iloc[-(SMA_PERIOD-1):].tolist()
            new_row['SMA_150_ADJ_CLOSE'] = (sum(last_149_adj_close) + tradier_quote['close']) / SMA_PERIOD
            new_row['SMA_PREV'] = hist.iloc[-SMA_SLOPE_PERIOD]['SMA_150_ADJ_CLOSE'] if len(hist) >= SMA_SLOPE_PERIOD else new_row['SMA_150_ADJ_CLOSE']
            new_row['AVG_SLOPE'] = ((new_row['SMA_150_ADJ_CLOSE'] / new_row['SMA_PREV']) - 1) * 100 if new_row['SMA_PREV'] else 0
            new_row['ADJ_CLOSE_PREV_5'] = prev_5_adj_close
            new_row['5_DAY_RISE'] = ((new_row['Adj_Close'] / prev_5_adj_close) - 1) * 100 if prev_5_adj_close else 0
            new_row['ABOVE_AVG_PCT'] = ((new_row['Adj_Close'] / new_row['SMA_150_ADJ_CLOSE']) - 1) * 100 if new_row['SMA_150_ADJ_CLOSE'] else 0
            last_row = new_row
            used_tradier = True
        else:
            used_tradier = False
    else:
        used_tradier = False
    def safe_val(val, fmt=None, dec=None):
        if pd.isna(val):
            return None
        if fmt == 'pct':
            return f"{round(val, 3):.3f}%"
        if dec is not None:
            return round(float(val), dec)
        if hasattr(val, 'item'):
            return val.item()
        return val
    key_order = [
        ('adj_close', safe_val(last_row['Adj_Close'], dec=5)),
        ('close', safe_val(last_row['Close'], dec=5)),
        ('5_day_rise', safe_val(last_row['5_DAY_RISE'], fmt='pct')),
        ('10_day_avg_slope', safe_val(last_row['AVG_SLOPE'], fmt='pct')),
        ('adj_price_above_avg_pct', safe_val(last_row['ABOVE_AVG_PCT'], fmt='pct')),
        ('sma150_adj_close', safe_val(last_row['SMA_150_ADJ_CLOSE'], dec=4)),
        ('Split', safe_val(last_row['Split_Ratio']))
    ]
    date_metrics = OrderedDict((k, v) for k, v in key_order if v is not None)
    date_str = str(last_row['Date'])
    if ticker_symbol == "SPY":
        SPY500_dict[date_str] = date_metrics
    if used_tradier:
        print(f"[OK+T] Processed {ticker_symbol} for {date_str} (+Tradier)")
    else:
        print(f"[OK] Processed {ticker_symbol} for {date_str}")
    return {date_str: date_metrics}


# --- Filter stocks against rules ---

def parse_pct(value):
    """Parse percentage string to float"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.replace('%', '').replace('$', ''))
    return None

def test_stock_against_rules(stock_data, rules):
    """Test a stock's data against the underlying_stock rules."""
    failed_rules = []
    underlying = rules.get("underlying_stock", {})
    
    # Get stock values
    adj_close = stock_data.get("adj_close")
    five_day_rise = parse_pct(stock_data.get("5_day_rise"))
    avg_slope = parse_pct(stock_data.get("10_day_avg_slope"))
    above_avg_pct = parse_pct(stock_data.get("adj_price_above_avg_pct"))
    
    # Get rule thresholds
    min_5_day_rise = parse_pct(underlying.get("min_5_day_rise_pct"))
    min_above_avg = parse_pct(underlying.get("min_above_avg_pct"))
    max_above_avg = parse_pct(underlying.get("max_above_avg_pct"))
    min_avg_slope = parse_pct(underlying.get("min_avg_up_slope_pct"))
    min_stock_price = parse_pct(underlying.get("min_stock_price"))
    
    # Test rules
    if min_5_day_rise is not None and five_day_rise is not None:
        if five_day_rise < min_5_day_rise:
            failed_rules.append("rl_Min_5_day_Rise")
    
    if min_above_avg is not None and above_avg_pct is not None:
        if above_avg_pct < min_above_avg:
            failed_rules.append("rlMinAboveAvg")
    
    if max_above_avg is not None and above_avg_pct is not None:
        if above_avg_pct > max_above_avg:
            failed_rules.append("rlMaxAboveAvg")
    
    if min_avg_slope is not None and avg_slope is not None:
        if avg_slope < min_avg_slope:
            failed_rules.append("rlAvgSlope")
    
    if min_stock_price is not None and adj_close is not None:
        if adj_close < min_stock_price:
            failed_rules.append("rlMinStockPrice")
    
    return len(failed_rules) == 0, failed_rules

def filter_stocks(rules_file, stocks_dict, output_file):
    """Filter stocks based on rules and output passing stocks"""
    with open(rules_file, 'r') as f:
        rules = json.load(f)
    passing_stocks = {}
    # Get the last SPY500 date
    spy_dates = sorted(SPY500_dict.keys())
    last_spy_date = spy_dates[-1] if spy_dates else None
    for ticker, ticker_data in stocks_dict.items():
        latest_date = list(ticker_data.keys())[0]
        # Filter out stocks missing the last SPY500 date
        if last_spy_date is None or latest_date != last_spy_date:
            continue
        stock_data = ticker_data[latest_date]
        passes, _ = test_stock_against_rules(stock_data, rules)
        if passes:
            passing_stocks[ticker] = {
                "date": latest_date,
                "adj_close": stock_data.get("adj_close"),
                "close": stock_data.get("close"),
                "5_day_rise": stock_data.get("5_day_rise"),
                "10_day_avg_slope": stock_data.get("10_day_avg_slope"),
                "adj_price_above_avg_pct": stock_data.get("adj_price_above_avg_pct"),
                "sma150_adj_close": stock_data.get("sma150_adj_close")
            }
    with open(output_file, 'w') as f:
        json.dump(passing_stocks, f, indent=4)
    print(f"[OK] {output_file}: {len(passing_stocks)} stocks pass rules")
    return passing_stocks

def filter_puts(result_file, rules_file):
    """
    Filter puts for each stock ticker in result file.
    
    Args:
        result_file: Path to result JSON file (e.g., 'result1.json')
        rules_file: Path to rules JSON file (e.g., 'rules1.json')
    
    Returns:
        Updated results dictionary
    """
    from datetime import datetime, timedelta
    
    # Load result file
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Load rules file
    with open(rules_file, 'r') as f:
        rules = json.load(f)
    
    # Get expiration rules
    entry_rules = rules.get("entry_put_position", {})
    min_days = int(entry_rules.get("min_days_for_expiration", "0"))
    max_days = int(entry_rules.get("max_days_for_expiration", "9999"))
    
    today = datetime.now().date()
    
    # Process each stock ticker
    for stock_ticker, stock_data in results.items():
        stock_data["puts"] = []
        
        # Loop through each expiration date
        for expiration_date in ExpirationDates:
            # Calculate days to expiration
            exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
            
            # Skip if outside rules
            if days_to_exp < min_days or days_to_exp > max_days:
                continue
            
            # Get put chain for this stock and expiration
            puts = get_put_chain(stock_ticker, expiration_date)
            
            # Add puts to stock data
            for put in puts:
                put["days_to_expiration"] = days_to_exp
                stock_data["puts"].append(put)
        
        print(f"[INFO] {stock_ticker}: {len(stock_data['puts'])} puts")
    
    # Save updated results
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[OK] Updated {result_file} with puts for {len(results)} tickers")
    return results


# --- Concurrent processing helper ---

TICKER_GROUP = 6  # Number of concurrent workers
DELAY_BETWEEN_WORKERS = 0.5  # Seconds delay between starting workers

def run_get_puts(stock_ticker):
    """Run get_puts.py for a single ticker and wait for completion"""
    try:
        result = subprocess.run(
            ["python", "get_puts.py", stock_ticker],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per ticker
        )
        if result.returncode == 0:
            # Extract put count from stdout (format: [RESULT] TICKER COUNT)
            put_count = 0
            for line in result.stdout.split('\n'):
                if line.startswith('[RESULT]'):
                    parts = line.split()
                    if len(parts) >= 3 and parts[2].isdigit():
                        put_count = int(parts[2])
            return stock_ticker, True, put_count
        else:
            return stock_ticker, False, result.stderr
    except subprocess.TimeoutExpired:
        return stock_ticker, False, "Timeout"
    except Exception as e:
        return stock_ticker, False, str(e)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Step 1: Generate rules_all.json with extreme values
generate_rules_all()

# Load tickers
with open('tickers.json', 'rb') as f:
    tickers = orjson.loads(f.read())

print(f"\n{'='*60}")
print(f"[INFO] Processing {len(tickers)} tickers with {TICKER_GROUP} parallel workers")
print(f"{'='*60}")

successful_tickers = []
failed_tickers = []
skipped_tickers = []

# Process tickers in groups
for i in range(0, len(tickers), TICKER_GROUP):
    group = tickers[i:i + TICKER_GROUP]
    print(f"\n[GROUP] Processing group {i//TICKER_GROUP + 1}/{(len(tickers) + TICKER_GROUP - 1)//TICKER_GROUP}: {', '.join(group)}")
    
    with ThreadPoolExecutor(max_workers=TICKER_GROUP) as executor:
        futures = {}
        for idx, ticker in enumerate(group):
            if idx > 0:
                time.sleep(DELAY_BETWEEN_WORKERS)
            futures[executor.submit(run_get_puts, ticker)] = ticker
        
        for future in as_completed(futures):
            ticker, success, result_info = future.result()
            if success:
                successful_tickers.append(ticker)
                print(f"  [OK] {ticker}: {result_info} puts")
            elif isinstance(result_info, str) and ("insufficient data" in result_info or "failed stock rules" in result_info or "No expiration dates" in result_info):
                skipped_tickers.append(ticker)
                print(f"  [SKIP] {ticker}")
            else:
                failed_tickers.append(ticker)
                print(f"  [FAIL] {ticker}: {result_info}")

print(f"\n{'='*60}")
print(f"[SUMMARY] Processing complete:")
print(f"   Successful: {len(successful_tickers)} tickers")
print(f"   Skipped (filtered out): {len(skipped_tickers)} tickers")
print(f"   Failed: {len(failed_tickers)} tickers")
if failed_tickers:
    print(f"   Failed tickers: {', '.join(failed_tickers)}")
print(f"{'='*60}")

# Calculate and print runtime
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")