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
    # Exclude rules_all.json (output)
    rules_files = [f for f in rules_files if f != "rules_all.json"]
    
    if not rules_files:
        print("[WARN] No rules files found")
        return
    
    # If only rules.json exists, just copy it
    if rules_files == ["rules.json"]:
        with open("rules.json", 'r') as f:
            rules = json.load(f)
        with open("rules_all.json", 'w') as f:
            json.dump(rules, f, indent=4)
        print(f"[OK] Generated rules_all.json from rules.json")
        return rules
    
    # Multiple rules files - merge with extreme values
    # Exclude rules.json if other rules files exist
    rules_files = [f for f in rules_files if f != "rules.json"]
    
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
        # Special cases: position_stop_loss is a max rule (higher = less restrictive)
        if 'min_' in key or key.startswith('min'):
            result = min(parsed)
        elif 'max_' in key or key.startswith('max') or 'stop_loss' in key:
            result = max(parsed)
        else:
            result = parsed[0]  # Non min/max rules - take first
        
        # Find original format
        for orig in original_vals:
            if orig is not None:
                return format_val(result, orig)
        return result
    
    # Build rules_all from extremes
    # Collect ALL sections and keys from ALL rules files
    rules_all = {}
    all_sections = set()
    for r in all_rules:
        all_sections.update(r.keys())
    
    for section in all_sections:
        # Find first file that has this section to use as template
        section_template = None
        for r in all_rules:
            if section in r and isinstance(r[section], dict):
                section_template = r[section]
                break
        
        if section_template is None:
            # Non-dict section, take from first file that has it
            for r in all_rules:
                if section in r:
                    rules_all[section] = r[section]
                    break
            continue
        
        # Collect all keys from all files for this section
        all_keys = set()
        for r in all_rules:
            if section in r and isinstance(r[section], dict):
                all_keys.update(r[section].keys())
        
        rules_all[section] = {}
        for key in all_keys:
            values = [r.get(section, {}).get(key) for r in all_rules]
            # Get first non-None value as template
            template_val = next((v for v in values if v is not None), None)
            
            if any(isinstance(v, bool) for v in values if v is not None):
                rules_all[section][key] = template_val  # Keep booleans as-is
            elif any(isinstance(v, str) and not any(c in v for c in '%$0123456789.-') for v in values if v):
                rules_all[section][key] = template_val  # Keep non-numeric strings as-is
            else:
                rules_all[section][key] = get_extreme(key, values, values)
    
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


def convert_timestamp_to_local(ts_ms, tz_offset=-6):
    """Convert Unix timestamp (milliseconds) to local time string 'yyyy-mm-dd hh:mm'"""
    if ts_ms is None:
        return None
    try:
        from datetime import datetime, timedelta, timezone
        # Convert milliseconds to seconds
        ts_sec = ts_ms / 1000
        # Convert to UTC datetime (timezone-aware)
        utc_dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        # Apply timezone offset
        local_dt = utc_dt + timedelta(hours=tz_offset)
        return local_dt.strftime("%Y-%m-%d %H:%M")
    except:
        return None


def test_put_against_high_mode(put, stock_data, rules):
    """Test a put against high mode (entry_put_position) rules. Returns True if passes."""
    entry_rules = rules.get("entry_put_position", {})
    
    # Get put values
    bid = put.get("bid", 0)
    delta = put.get("delta")
    ask = put.get("ask", 0)
    strike = put.get("strike", 0)
    days_to_exp = put.get("days_to_expiration", 0)
    adj_close = stock_data.get("adj_close", 0)
    
    # Get rule thresholds
    min_bid = parse_pct(entry_rules.get("min_put_bid_price", "0"))
    min_delta = parse_pct(entry_rules.get("min_put_delta", "-100"))
    max_delta = parse_pct(entry_rules.get("max_put_delta", "0"))
    max_ask_above_bid = parse_pct(entry_rules.get("max_ask_above_bid_pct", "100"))
    min_avg_above_strike = parse_pct(entry_rules.get("min_avg_above_strike", "-100"))
    min_rr_ratio = parse_pct(entry_rules.get("min_risk_reward_ratio", "-1000000"))
    min_annual_rr = parse_pct(entry_rules.get("min_annual_risk_reward_ratio", "-1000000"))
    min_rev_annual_rr = parse_pct(entry_rules.get("min_rev_annual_rr_ratio", "-1000000"))
    max_days = int(float(entry_rules.get("max_days_for_expiration", "9999")))
    
    # Test rules
    if min_bid is not None and bid < min_bid:
        return False
    
    if delta is not None:
        delta_pct = delta * 100 if abs(delta) <= 1 else delta
        if min_delta is not None and delta_pct < min_delta:
            return False
        if max_delta is not None and delta_pct > max_delta:
            return False
    
    if bid > 0 and ask > 0:
        ask_above_bid_pct = ((ask - bid) / bid) * 100
        if max_ask_above_bid is not None and ask_above_bid_pct > max_ask_above_bid:
            return False
    
    if strike > 0 and adj_close > 0:
        avg_above_strike_pct = ((adj_close - strike) / strike) * 100
        if min_avg_above_strike is not None and avg_above_strike_pct < min_avg_above_strike:
            return False
    
    if days_to_exp > max_days:
        return False
    
    # Risk/reward ratio tests
    if bid > 0 and strike > bid and days_to_exp > 0:
        risk_reward_ratio = -((strike - bid) / bid)
        if min_rr_ratio is not None and risk_reward_ratio < min_rr_ratio:
            return False
        
        annual_rr = risk_reward_ratio * (365.0 / days_to_exp)
        if min_annual_rr is not None and annual_rr < min_annual_rr:
            return False
        
        rev_annual_rr = risk_reward_ratio * (days_to_exp / 365.0)
        if min_rev_annual_rr is not None and rev_annual_rr < min_rev_annual_rr:
            return False
    
    return True


def test_put_against_low_mode(put, stock_data, rules):
    """Test a put against low mode (low_put_mode) rules. Returns True if passes."""
    low_rules = rules.get("low_put_mode", {})
    
    if not low_rules:
        return False  # No low mode rules defined
    
    # Get put values
    bid = put.get("bid", 0)
    delta = put.get("delta")
    ask = put.get("ask", 0)
    strike = put.get("strike", 0)
    days_to_exp = put.get("days_to_expiration", 0)
    adj_close = stock_data.get("adj_close", 0)
    
    # Get rule thresholds (low mode uses "low_" prefix)
    min_bid = parse_pct(low_rules.get("low_min_put_bid_price", "0"))
    min_delta = parse_pct(low_rules.get("low_min_put_delta", "-100"))
    max_delta = parse_pct(low_rules.get("low_max_put_delta", "0"))
    max_ask_above_bid = parse_pct(low_rules.get("low_max_ask_above_bid_pct", "100"))
    min_rr_ratio = parse_pct(low_rules.get("low_min_risk_reward_ratio", "-1000000"))
    min_annual_rr = parse_pct(low_rules.get("low_min_annual_risk_reward_ratio", "-1000000"))
    min_rev_annual_rr = parse_pct(low_rules.get("low_min_rev_annual_rr_ratio", "-1000000"))
    max_days = int(float(low_rules.get("low_max_days_for_expiration", "9999")))
    
    # Test rules
    if min_bid is not None and bid < min_bid:
        return False
    
    if delta is not None:
        delta_pct = delta * 100 if abs(delta) <= 1 else delta
        if min_delta is not None and delta_pct < min_delta:
            return False
        if max_delta is not None and delta_pct > max_delta:
            return False
    
    if bid > 0 and ask > 0:
        ask_above_bid_pct = ((ask - bid) / bid) * 100
        if max_ask_above_bid is not None and ask_above_bid_pct > max_ask_above_bid:
            return False
    
    if days_to_exp > max_days:
        return False
    
    # Risk/reward ratio tests
    if bid > 0 and strike > bid and days_to_exp > 0:
        risk_reward_ratio = -((strike - bid) / bid)
        if min_rr_ratio is not None and risk_reward_ratio < min_rr_ratio:
            return False
        
        annual_rr = risk_reward_ratio * (365.0 / days_to_exp)
        if min_annual_rr is not None and annual_rr < min_annual_rr:
            return False
        
        rev_annual_rr = risk_reward_ratio * (days_to_exp / 365.0)
        if min_rev_annual_rr is not None and rev_annual_rr < min_rev_annual_rr:
            return False
    
    return True


def test_stock_against_low_mode(stock_data, rules):
    """Test a stock against low mode (low_put_mode) stock rules. Returns True if passes."""
    low_rules = rules.get("low_put_mode", {})
    
    if not low_rules:
        return False  # No low mode rules defined
    
    # Get stock values
    adj_close = stock_data.get("adj_close")
    five_day_rise = parse_pct(stock_data.get("5_day_rise"))
    avg_slope = parse_pct(stock_data.get("10_day_avg_slope"))
    above_avg_pct = parse_pct(stock_data.get("adj_price_above_avg_pct"))
    
    # Get rule thresholds (low mode uses "low_" prefix)
    min_5_day_rise = parse_pct(low_rules.get("low_min_5_day_rise_pct"))
    min_above_avg = parse_pct(low_rules.get("low_min_above_avg_pct"))
    max_above_avg = parse_pct(low_rules.get("low_max_above_avg_pct"))
    min_avg_slope = parse_pct(low_rules.get("low_min_avg_up_slope_pct"))
    min_stock_price = parse_pct(low_rules.get("low_min_stock_price"))
    
    # Test rules
    if min_5_day_rise is not None and five_day_rise is not None:
        if five_day_rise < min_5_day_rise:
            return False
    
    if min_above_avg is not None and above_avg_pct is not None:
        if above_avg_pct < min_above_avg:
            return False
    
    if max_above_avg is not None and above_avg_pct is not None:
        if above_avg_pct > max_above_avg:
            return False
    
    if min_avg_slope is not None and avg_slope is not None:
        if avg_slope < min_avg_slope:
            return False
    
    if min_stock_price is not None and adj_close is not None:
        if adj_close < min_stock_price:
            return False
    
    return True


def generate_results_from_put_data():
    """Generate result1.json through result6.json by filtering put_data_today/ against each rules file"""
    import os
    
    # Load all put_data_today files
    put_data_dir = "put_data_today"
    if not os.path.exists(put_data_dir):
        print(f"[WARN] {put_data_dir} directory not found")
        return
    
    all_put_data = {}
    for filename in os.listdir(put_data_dir):
        if filename.endswith('.json'):
            ticker = filename.replace('.json', '')
            filepath = os.path.join(put_data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if data.get("status") == "success":
                        all_put_data[ticker] = data
            except Exception as e:
                print(f"[WARN] Error loading {filepath}: {e}")
    
    print(f"[INFO] Loaded {len(all_put_data)} tickers from put_data_today/")
    
    # Process each rules file
    for i in range(1, 7):
        rules_file = f"rules{i}.json"
        result_file = f"result{i}.json"
        
        if not os.path.exists(rules_file):
            print(f"[SKIP] {rules_file} not found")
            continue
        
        with open(rules_file, 'r') as f:
            rules = json.load(f)
        
        results = {}
        total_puts = 0
        
        for ticker, data in all_put_data.items():
            stock_data = data.get("stock_data", {})
            puts = data.get("puts", [])
            
            # Test stock against high mode (underlying_stock) rules
            passes_stock_high = test_stock_against_rules(stock_data, rules)[0]
            
            # Test stock against low mode rules
            passes_stock_low = test_stock_against_low_mode(stock_data, rules)
            
            # If stock fails both modes, skip entirely
            if not passes_stock_high and not passes_stock_low:
                continue
            
            # Filter puts against this rules file's high and low modes
            filtered_puts = []
            tz_offset = -time.timezone // 3600  # Calculate local timezone offset
            
            for put in puts:
                # Convert bid_date and ask_date to local time format
                if put.get("bid_date") and isinstance(put.get("bid_date"), (int, float)):
                    put["bid_date"] = convert_timestamp_to_local(put["bid_date"], tz_offset)
                if put.get("ask_date") and isinstance(put.get("ask_date"), (int, float)):
                    put["ask_date"] = convert_timestamp_to_local(put["ask_date"], tz_offset)
                
                # Test put against both modes
                passes_high = passes_stock_high and test_put_against_high_mode(put, stock_data, rules)
                passes_low = passes_stock_low and test_put_against_low_mode(put, stock_data, rules)
                
                # If put fails both modes, skip it
                if not passes_high and not passes_low:
                    continue
                
                # Add mode pass flags to put
                put["passed_high_put_mode"] = passes_high
                put["passed_low_put_mode"] = passes_low
                filtered_puts.append(put)
            
            if filtered_puts:
                results[ticker] = {
                    "date": stock_data.get("date"),
                    "adj_close": stock_data.get("adj_close"),
                    "close": stock_data.get("close"),
                    "5_day_rise": stock_data.get("5_day_rise"),
                    "10_day_avg_slope": stock_data.get("10_day_avg_slope"),
                    "adj_price_above_avg_pct": stock_data.get("adj_price_above_avg_pct"),
                    "sma150_adj_close": stock_data.get("sma150_adj_close"),
                    "puts": filtered_puts
                }
                total_puts += len(filtered_puts)
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"[OK] {result_file}: {len(results)} stocks, {total_puts} puts")


def generate_result_all():
    """Generate result_all.json with unique puts ranked by rev_annual_rr_ratio (highest first)"""
    
    # First pass: collect which rules files each put passes for low and high modes
    put_rules_map = {}  # symbol -> {"low": [rules files], "high": [rules files], "put_data": {...}}
    
    for i in range(1, 7):
        result_file = f"result{i}.json"
        rules_file = f"rules{i}.json"
        if not os.path.exists(result_file):
            continue
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        for stock_ticker, stock_data in results.items():
            puts = stock_data.get("puts", [])
            for put in puts:
                symbol = put.get("symbol")
                if not symbol:
                    continue
                
                if symbol not in put_rules_map:
                    # Rename "symbol" to "put_symbol" in put_data
                    put_data = {k: v for k, v in put.items() if k not in ["passed_high_put_mode", "passed_low_put_mode"]}
                    if "symbol" in put_data:
                        put_data["put_symbol"] = put_data.pop("symbol")
                    
                    put_rules_map[symbol] = {
                        "low": [],
                        "high": [],
                        "stock_ticker": stock_ticker,
                        "stock_date": stock_data.get("date"),
                        "stock_close": stock_data.get("close"),
                        "stock_adj_close": stock_data.get("adj_close"),
                        "stock_5_day_rise": stock_data.get("5_day_rise"),
                        "stock_10_day_avg_slope": stock_data.get("10_day_avg_slope"),
                        "stock_adj_price_above_avg_pct": stock_data.get("adj_price_above_avg_pct"),
                        "stock_sma150_adj_close": stock_data.get("sma150_adj_close"),
                        "put_data": put_data
                    }
                
                # Track which rules files this put passed for each mode
                if put.get("passed_low_put_mode"):
                    if rules_file not in put_rules_map[symbol]["low"]:
                        put_rules_map[symbol]["low"].append(rules_file)
                if put.get("passed_high_put_mode"):
                    if rules_file not in put_rules_map[symbol]["high"]:
                        put_rules_map[symbol]["high"].append(rules_file)
    
    # Build final list, filtering out puts that passed neither mode in any rules file
    all_puts = []
    for symbol, data in put_rules_map.items():
        # If both low and high lists are empty, skip this put
        if not data["low"] and not data["high"]:
            continue
        
        # Extract put_symbol from put_data to place it after stock fields
        put_data = data["put_data"]
        put_symbol = put_data.pop("put_symbol", None)
        
        put_entry = {
            "stock_ticker": data["stock_ticker"],
            "stock_date": data["stock_date"],
            "stock_close": data["stock_close"],
            "stock_adj_close": data["stock_adj_close"],
            "stock_5_day_rise": data["stock_5_day_rise"],
            "stock_10_day_avg_slope": data["stock_10_day_avg_slope"],
            "stock_adj_price_above_avg_pct": data["stock_adj_price_above_avg_pct"],
            "stock_sma150_adj_close": data["stock_sma150_adj_close"],
            "put_symbol": put_symbol,
            **put_data,
            "source_low_rules": ", ".join(data["low"]) if data["low"] else "",
            "source_high_rules": ", ".join(data["high"]) if data["high"] else ""
        }
        all_puts.append(put_entry)
    
    # Sort by rev_annual_rr_ratio descending (highest first = best)
    # Puts without the ratio go to the end
    all_puts.sort(key=lambda x: x.get("rev_annual_rr_ratio") or float('-inf'), reverse=True)
    
    # Add rank_order (1 = top ranked put, 2 = second, etc.)
    for rank, put in enumerate(all_puts, start=1):
        put["rank_order"] = rank
    
    # Save to result_all.json
    with open("result_all.json", 'w') as f:
        json.dump(all_puts, f, indent=4)
    
    print(f"[OK] result_all.json: {len(all_puts)} unique puts (sorted by rev_annual_rr_ratio)")
    return all_puts


# --- Concurrent processing helper ---

TICKER_GROUP = 16  # Number of concurrent workers
DELAY_BETWEEN_WORKERS = 0.5  # Seconds delay between starting workers

def run_get_puts(stock_ticker):
    """Run get_puts_today.py for a single ticker and wait for completion"""
    try:
        result = subprocess.run(
            ["python", "get_puts_today.py", stock_ticker],
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

if __name__ == "__main__":
    # Step 1: Generate rules_all.json with extreme values
    generate_rules_all()

    # Step 2: Clear put_data_today/ folder to start fresh
    put_data_dir = "put_data_today"
    if os.path.exists(put_data_dir):
        import shutil
        shutil.rmtree(put_data_dir)
        print(f"[OK] Cleared {put_data_dir}/ folder")
    os.makedirs(put_data_dir, exist_ok=True)

    # Load tickers
    with open('tickers.json', 'rb') as f:
        tickers = orjson.loads(f.read())

    # Timezone info (calculated UTC offset in hours)
    time_zone = -time.timezone // 3600  # -6 for CST

    print(f"\n{'='*60}")
    print(f"[INFO] Processing {len(tickers)} tickers with {TICKER_GROUP} parallel workers")
    print(f"{'='*60}")

    successful_tickers = []
    failed_tickers = []
    skipped_tickers = []
    total_puts = 0

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
                    total_puts += result_info
                    print(f"  [OK] {ticker}: {result_info} puts")
                elif isinstance(result_info, str) and ("insufficient data" in result_info or "failed stock rules" in result_info or "No expiration dates" in result_info):
                    skipped_tickers.append(ticker)
                    print(f"  [SKIP] {ticker}")
                else:
                    failed_tickers.append(ticker)
                    print(f"  [FAIL] {ticker}: {result_info}")
        
        print(f"  [TOTAL] {total_puts} puts so far")

    print(f"\n{'='*60}")
    print(f"[SUMMARY] Processing complete:")
    print(f"   Successful: {len(successful_tickers)} tickers")
    print(f"   Skipped (filtered out): {len(skipped_tickers)} tickers")
    print(f"   Failed: {len(failed_tickers)} tickers")
    print(f"   Total puts: {total_puts}")
    if failed_tickers:
        print(f"   Failed tickers: {', '.join(failed_tickers)}")
    print(f"{'='*60}")

    # Generate result1.json through result6.json from put_data_today/
    print(f"\n[INFO] Generating result files from put_data_today/...")
    generate_results_from_put_data()

    # Generate result_all.json with unique puts ranked by rev_annual_rr_ratio
    print(f"\n[INFO] Generating result_all.json...")
    generate_result_all()

    # Calculate and print runtime
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")