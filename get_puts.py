"""
get_puts.py - Get filtered puts for a single stock ticker

Usage:
    python get_puts.py ABNB
    python get_puts.py AAPL

This script:
1. Downloads Yahoo data for the ticker
2. Filters stock against rules_all.json (underlying_stock rules)
3. If stock passes, gets put options filtered by entry_put_position rules
4. Saves result to put_data/{ticker}.json
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import requests
import json
import os
import sys
import time

# Tradier API configuration
TRADIER_API_KEY = "6IyR6wDsuQ2tzm9mGxzLDQY1GrTF"
TRADIER_BASE_URL = "https://api.tradier.com/v1"

# Configuration
SMA_PERIOD = 150
SMA_SLOPE_PERIOD = 10
RISE_PERIOD = 5

# Define date range
start_date = (datetime.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%Y-%m-%d")


def get_tradier_quote(stock_ticker):
    """Get current quote from Tradier API"""
    # Handle special ticker symbols (Tradier uses forward slash)
    ticker = stock_ticker
    if stock_ticker in ["BF.B", "BF-B"]:
        ticker = "BF/B"
    elif stock_ticker in ["BRK.B", "BRK-B"]:
        ticker = "BRK/B"
    elif stock_ticker == "FI":
        ticker = "FISV"
    
    url = f"{TRADIER_BASE_URL}/markets/quotes?symbols={ticker}"
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
        print(f"Tradier error for {stock_ticker}: {e}")
    return None


def get_option_expirations(stock_ticker):
    """Get all available option expiration dates from Tradier API"""
    # Handle special ticker symbols
    ticker = stock_ticker
    if stock_ticker in ["BF.B", "BF-B"]:
        ticker = "BF/B"
    elif stock_ticker in ["BRK.B", "BRK-B"]:
        ticker = "BRK/B"
    elif stock_ticker == "FI":
        ticker = "FISV"
    
    url = f"{TRADIER_BASE_URL}/markets/options/expirations?symbol={ticker}"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "expirations" in data and "date" in data["expirations"]:
                return data["expirations"]["date"]
    except Exception as e:
        print(f"Tradier expiration error for {stock_ticker}: {e}")
    return []


def get_put_chain(stock_ticker, expiration_date):
    """Get all put options for a stock ticker and expiration date from Tradier API."""
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


def process_ticker(stock_ticker):
    """Download and process Yahoo data for a single ticker"""
    yf_ticker = yf.Ticker(stock_ticker)
    hist_original = yf_ticker.history(start=start_date, end=end_date)
    
    if len(hist_original) < 160:
        print(f"[SKIP] {stock_ticker} (insufficient data: {len(hist_original)} days)")
        return None
    
    hist = hist_original.copy()
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index)
    
    dividends = yf_ticker.dividends
    splits = yf_ticker.splits
    hist['Dividends'] = dividends
    hist['Stock Splits'] = splits
    hist['Dividends'] = hist['Dividends'].fillna(0)
    hist['Stock Splits'] = hist['Stock Splits'].fillna(0)
    hist['Adj_Close'] = hist['Close']
    
    # Adjust for splits
    split_dates = hist[hist['Stock Splits'] > 0].index
    for split_date in split_dates:
        split_ratio = hist.loc[split_date, 'Stock Splits']
        hist.loc[:split_date - pd.Timedelta(days=1), ['Close', 'High', 'Low', 'Open']] *= split_ratio
    
    # Adjust for dividends
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
    used_tradier = False
    
    # If Yahoo data is stale, get fresh quote from Tradier
    if yahoo_date != today_str:
        tradier_quote = get_tradier_quote(stock_ticker)
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
    
    if used_tradier:
        print(f"[OK+T] Processed {stock_ticker} for {date_str} (+Tradier)")
    else:
        print(f"[OK] Processed {stock_ticker} for {date_str}")
    
    return {"date": date_str, **date_metrics}


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


def test_put_against_rules(put_data, stock_data, rules):
    """Test a put option against the entry_put_position rules."""
    failed_rules = []
    entry_rules = rules.get("entry_put_position", {})
    
    # Get put values
    bid = put_data.get("bid") or 0
    ask = put_data.get("ask") or 0
    delta = put_data.get("delta")
    strike = put_data.get("strike") or 0
    days_to_exp = put_data.get("days_to_expiration") or 0
    
    # Get stock values
    adj_close = stock_data.get("adj_close") or 0
    sma150 = stock_data.get("sma150_adj_close") or adj_close
    
    # Get rule thresholds
    min_days = int(float(entry_rules.get("min_days_for_expiration", "0")))
    max_days = int(float(entry_rules.get("max_days_for_expiration", "9999")))
    min_bid = parse_pct(entry_rules.get("min_put_bid_price", "0"))
    min_delta = parse_pct(entry_rules.get("min_put_delta", "-100"))
    max_delta = parse_pct(entry_rules.get("max_put_delta", "0"))
    max_ask_above_bid = parse_pct(entry_rules.get("max_ask_above_bid_pct", "100"))
    min_avg_above_strike = parse_pct(entry_rules.get("min_avg_above_strike", "-100"))
    
    # Risk/reward rule thresholds
    min_rr_ratio = parse_pct(entry_rules.get("min_risk_reward_ratio", "-1000000"))
    min_annual_rr = parse_pct(entry_rules.get("min_annual_risk_reward_ratio", "-1000000"))
    min_rev_annual_rr = parse_pct(entry_rules.get("min_rev_annual_rr_ratio", "-1000000"))
    min_expected_profit = parse_pct(entry_rules.get("min_expected_profit", "-1000000")) / 100  # Convert from % to decimal
    
    # Test rules
    if days_to_exp < min_days:
        failed_rules.append("rlMinDaysExp")
    
    if days_to_exp > max_days:
        failed_rules.append("rlMaxDaysExp")
    
    if min_bid is not None and bid < min_bid:
        failed_rules.append("rlMinBid")
    
    # Delta checks (convert to percentage if needed)
    delta_pct = None
    if delta is not None:
        delta_pct = delta * 100 if abs(delta) <= 1 else delta
        if min_delta is not None and delta_pct < min_delta:
            failed_rules.append("rlMinDelta")
        if max_delta is not None and delta_pct > max_delta:
            failed_rules.append("rlMaxDelta")
    
    # Ask above bid percentage
    if bid > 0:
        ask_above_bid_pct = ((ask - bid) / bid) * 100
        if max_ask_above_bid is not None and ask_above_bid_pct > max_ask_above_bid:
            failed_rules.append("rlMaxAskAboveBid")
    
    # Average above strike percentage
    if strike > 0:
        avg_above_strike_pct = ((sma150 - strike) / strike) * 100
        if min_avg_above_strike is not None and avg_above_strike_pct < min_avg_above_strike:
            failed_rules.append("rlMinAvgAboveStrike")
    
    # --- Risk/Reward Ratio calculations (from sim.py) ---
    # risk_reward_ratio = -(strike - bid) / bid
    # annual_risk = risk_reward_ratio * (365 / days_to_exp)
    # rev_annual_risk = risk_reward_ratio * (days_to_exp / 365)
    # expected_profit = (bid * (1 + delta) + (strike - bid) * delta) / bid
    
    if bid > 0 and strike > bid:
        # Calculate risk_reward_ratio: -(Risk / Reward) = -(Strike - Bid) / Bid
        risk_reward_ratio = -((strike - bid) / bid)
        
        if min_rr_ratio is not None and risk_reward_ratio < min_rr_ratio:
            failed_rules.append("rlMinRiskReward")
        
        # Calculate annual_risk = risk_reward_ratio * (365 / DTE)
        if days_to_exp > 0:
            annual_risk = risk_reward_ratio * (365.0 / days_to_exp)
            if min_annual_rr is not None and annual_risk < min_annual_rr:
                failed_rules.append("rlMinAnnualRisk")
            
            # Calculate rev_annual_risk = risk_reward_ratio * (DTE / 365)
            rev_annual_risk = risk_reward_ratio * (days_to_exp / 365.0)
            if min_rev_annual_rr is not None and rev_annual_risk < min_rev_annual_rr:
                failed_rules.append("rlMinRevAnnualRisk")
        
        # Calculate expected_profit using delta
        # expected_profit = (bid * (1 + delta) + (strike - bid) * delta) / bid
        if delta is not None:
            delta_decimal = delta if abs(delta) <= 1 else delta / 100
            expected_profit_calc = (bid * (1.0 + delta_decimal) + (strike - bid) * delta_decimal) / bid
            if min_expected_profit is not None and expected_profit_calc < min_expected_profit:
                failed_rules.append("rlMinExpectedProfit")
    
    return len(failed_rules) == 0, failed_rules


def get_puts_for_ticker(stock_ticker, rules_file="rules_all.json"):
    """
    Main function to get filtered puts for a single stock ticker.
    
    Args:
        stock_ticker: Stock symbol (e.g., 'ABNB')
        rules_file: Path to rules JSON file (default: 'rules_all.json')
    
    Returns:
        Result dictionary or None if stock doesn't pass rules
    """
    # Load rules
    with open(rules_file, 'r') as f:
        rules = json.load(f)
    
    # Step 1: Get stock data
    print(f"\n{'='*50}")
    print(f"[PROCESSING] {stock_ticker}")
    print(f"{'='*50}")
    
    # Create put_data directory if it doesn't exist
    os.makedirs("put_data", exist_ok=True)
    output_file = f"put_data/{stock_ticker}.json"
    
    stock_data = process_ticker(stock_ticker)
    if stock_data is None:
        # Save empty result for failed/delisted stocks
        result = {"ticker": stock_ticker, "status": "no_data", "puts": []}
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        return None
    
    # Step 2: Test stock against rules
    passes, failed_rules = test_stock_against_rules(stock_data, rules)
    if not passes:
        print(f"[FAIL] {stock_ticker} failed stock rules: {', '.join(failed_rules)}")
        # Save result with failure reason
        result = {
            "ticker": stock_ticker,
            "status": "failed_stock_rules",
            "failed_rules": failed_rules,
            "stock_data": stock_data,
            "puts": []
        }
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        return None
    
    print(f"[PASS] {stock_ticker} passed stock filter")
    
    # Step 3: Get expiration dates
    expiration_dates = get_option_expirations(stock_ticker)
    if not expiration_dates:
        print(f"[FAIL] No expiration dates found for {stock_ticker}")
        # Save result with no expirations
        result = {
            "ticker": stock_ticker,
            "status": "no_expirations",
            "stock_data": stock_data,
            "puts": []
        }
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        return None
    
    print(f"[INFO] Found {len(expiration_dates)} expiration dates")
    
    # Step 4: Get and filter puts
    entry_rules = rules.get("entry_put_position", {})
    min_days = int(float(entry_rules.get("min_days_for_expiration", "0")))
    max_days = int(float(entry_rules.get("max_days_for_expiration", "9999")))
    
    today = datetime.now().date()
    filtered_puts = []
    
    for exp_date_str in expiration_dates:
        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
        days_to_exp = (exp_date - today).days
        
        # Skip if outside day range
        if days_to_exp < min_days or days_to_exp > max_days:
            continue
        
        # Get put chain
        puts = get_put_chain(stock_ticker, exp_date_str)
        
        for put in puts:
            put["days_to_expiration"] = days_to_exp
            
            # Test put against rules
            passes, failed = test_put_against_rules(put, stock_data, rules)
            if passes:
                filtered_puts.append(put)
    
    print(f"[PUTS] {stock_ticker}: {len(filtered_puts)} puts pass filter")
    
    # Build result
    result = {
        "ticker": stock_ticker,
        "status": "success",
        "stock_data": stock_data,
        "puts": filtered_puts
    }
    
    # Save to put_data/{ticker}.json (directory already created at start)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"[SAVED] {output_file}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_puts.py <TICKER>")
        print("Example: python get_puts.py ABNB")
        sys.exit(1)
    
    start_time = time.time()
    
    ticker = sys.argv[1].upper()
    result = get_puts_for_ticker(ticker)
    
    if result:
        put_count = len(result.get("puts", []))
        print(f"\n[RESULT] {ticker} {put_count}")
    else:
        print(f"\n[RESULT] {ticker} FAIL")
    
    # Calculate and print runtime
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
