"""
refresh_results.py - Refresh market data for top puts in result_all.json

This script:
1. Reads result_all.json
2. Takes the top N puts (default 100) ranked by rev_annual_rr_ratio
3. Fetches fresh market data from Tradier API for each put
4. Recalculates risk/reward ratios
5. Re-ranks and saves back to result_all.json
"""

import json
import requests
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Tradier API configuration
TRADIER_API_KEY = "6IyR6wDsuQ2tzm9mGxzLDQY1GrTF"
TRADIER_BASE_URL = "https://api.tradier.com/v1"

# Configuration
TOP_N = 20  # Number of top puts to refresh
WORKERS = 8  # Number of concurrent API requests
DELAY_BETWEEN_WORKERS = 0.3  # Seconds delay between starting workers


def convert_timestamp_to_local(ts_ms, tz_offset=-6):
    """Convert Unix timestamp (milliseconds) to local time string 'yyyy-mm-dd hh:mm'"""
    if ts_ms is None:
        return None
    try:
        ts_sec = ts_ms / 1000
        utc_dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        local_dt = utc_dt + timedelta(hours=tz_offset)
        return local_dt.strftime("%Y-%m-%d %H:%M")
    except:
        return None


def get_stock_quote(ticker):
    """
    Get current stock quote from Tradier API.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary with close price or None if failed
    """
    # Handle special ticker symbols
    api_ticker = ticker
    if ticker in ["BF.B", "BF-B"]:
        api_ticker = "BF/B"
    elif ticker in ["BRK.B", "BRK-B"]:
        api_ticker = "BRK/B"
    elif ticker == "FI":
        api_ticker = "FISV"
    
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    params = {"symbols": api_ticker}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if "quotes" in data and "quote" in data["quotes"]:
                quote = data["quotes"]["quote"]
                return {
                    "close": quote.get("close") or quote.get("last"),
                    "last": quote.get("last")
                }
    except Exception as e:
        print(f"Error fetching stock quote for {ticker}: {e}")
    
    return None


def get_option_quote(put_symbol):
    """
    Get fresh quote for a single option symbol from Tradier API.
    
    Args:
        put_symbol: Option symbol (e.g., 'CRWD260821P00820000')
    
    Returns:
        Dictionary with fresh option data or None if failed
    """
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    
    # Parse the option symbol to extract underlying and expiration
    # Format: TICKER + YYMMDD + P/C + STRIKE (8 digits)
    # Example: CRWD260821P00820000
    try:
        # Find where the date starts (6 digits after ticker)
        # The ticker is everything before the date
        for i in range(len(put_symbol) - 15, 0, -1):
            if put_symbol[i:i+6].isdigit():
                ticker = put_symbol[:i]
                date_part = put_symbol[i:i+6]
                option_type = put_symbol[i+6]
                strike_part = put_symbol[i+7:]
                break
        else:
            return None
        
        # Convert date YYMMDD to YYYY-MM-DD
        year = "20" + date_part[:2]
        month = date_part[2:4]
        day = date_part[4:6]
        expiration_date = f"{year}-{month}-{day}"
        
        # Handle special ticker symbols
        api_ticker = ticker
        if ticker in ["BF.B", "BF-B"]:
            api_ticker = "BF/B"
        elif ticker in ["BRK.B", "BRK-B"]:
            api_ticker = "BRK/B"
        elif ticker == "FI":
            api_ticker = "FISV"
        
        # Get option chain for this expiration
        params = {
            "symbol": api_ticker,
            "expiration": expiration_date,
            "greeks": "true"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if "options" in data and data["options"] and "option" in data["options"]:
                # Find the matching option
                for option in data["options"]["option"]:
                    if option.get("symbol") == put_symbol:
                        greeks = option.get("greeks", {}) or {}
                        return {
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
                            "mid_iv": greeks.get("mid_iv")
                        }
    except Exception as e:
        print(f"Error parsing/fetching {put_symbol}: {e}")
    
    return None


def refresh_put(put_entry):
    """
    Refresh a single put entry with fresh market data.
    
    Args:
        put_entry: Dictionary with put data
    
    Returns:
        Updated put_entry or original if refresh failed
    """
    put_symbol = put_entry.get("put_symbol")
    if not put_symbol:
        return put_entry
    
    # Get fresh data from Tradier
    fresh_data = get_option_quote(put_symbol)
    
    if fresh_data is None:
        print(f"  [SKIP] {put_symbol} - No data")
        return put_entry
    
    # Calculate timezone offset
    tz_offset = -time.timezone // 3600
    
    # Update put entry with fresh data (only specified fields)
    put_entry["bid"] = fresh_data.get("bid")
    put_entry["ask"] = fresh_data.get("ask")
    
    # Convert timestamps to local time
    bid_date = fresh_data.get("bid_date")
    ask_date = fresh_data.get("ask_date")
    if bid_date and isinstance(bid_date, (int, float)):
        put_entry["bid_date"] = convert_timestamp_to_local(bid_date, tz_offset)
    elif bid_date:
        put_entry["bid_date"] = bid_date
    
    if ask_date and isinstance(ask_date, (int, float)):
        put_entry["ask_date"] = convert_timestamp_to_local(ask_date, tz_offset)
    elif ask_date:
        put_entry["ask_date"] = ask_date
    
    put_entry["last_price"] = fresh_data.get("last_price")
    
    # Format delta as percentage string
    delta_val = fresh_data.get("delta")
    put_entry["delta"] = f"{delta_val * 100:.1f}%" if delta_val is not None else None
    
    # Recalculate risk/reward ratios
    bid = put_entry.get("bid") or 0
    strike = put_entry.get("strike") or 0
    expiration_date = put_entry.get("expiration_date")
    
    # Calculate days to expiration
    days_to_exp = 0
    if expiration_date:
        try:
            exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            today = datetime.now()
            days_to_exp = (exp_date - today).days
        except:
            pass
    
    if bid > 0 and strike > bid and days_to_exp > 0:
        risk_reward_ratio = -((strike - bid) / bid)
        annual_rr_ratio = risk_reward_ratio * (365.0 / days_to_exp)
        rev_annual_rr_ratio = risk_reward_ratio * (days_to_exp / 365.0)
        
        put_entry["risk_reward_ratio"] = round(risk_reward_ratio, 6)
        put_entry["annual_rr_ratio"] = round(annual_rr_ratio, 6)
        put_entry["rev_annual_rr_ratio"] = round(rev_annual_rr_ratio, 6)
    
    # Recalculate intrinsic_value and time_val
    stock_close = put_entry.get("stock_close") or 0
    put_entry["intrinsic_value"] = round(max(0, strike - stock_close), 2)
    put_entry["time_val"] = round(bid + stock_close - strike, 2)
    
    print(f"  [OK] {put_symbol} - bid: {bid}")
    return put_entry


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("Refresh Results - Update top puts with fresh market data")
    print("=" * 60)
    
    # Load result_all.json
    print(f"\n[INFO] Loading result_all.json...")
    try:
        with open("result_all.json", "r") as f:
            all_puts = json.load(f)
    except FileNotFoundError:
        print("[ERROR] result_all.json not found")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in result_all.json: {e}")
        return
    
    print(f"[INFO] Loaded {len(all_puts)} puts")
    
    # Sort by rev_annual_rr_ratio descending (highest first = best)
    all_puts.sort(key=lambda x: x.get("rev_annual_rr_ratio") or float('-inf'), reverse=True)
    
    # Get top N puts to refresh
    top_puts = all_puts[:TOP_N]
    remaining_puts = all_puts[TOP_N:]
    
    # Step 1: Refresh unique stock prices first
    unique_tickers = list(set(put.get("stock_ticker") for put in top_puts if put.get("stock_ticker")))
    print(f"\n[INFO] Refreshing {len(unique_tickers)} unique stock prices...")
    
    stock_prices = {}
    for ticker in unique_tickers:
        quote = get_stock_quote(ticker)
        if quote and quote.get("close"):
            stock_prices[ticker] = quote["close"]
            print(f"  [OK] {ticker}: ${quote['close']}")
        else:
            print(f"  [SKIP] {ticker} - No quote")
    
    # Update stock prices in top_puts
    for put in top_puts:
        ticker = put.get("stock_ticker")
        if ticker in stock_prices:
            put["stock_close"] = stock_prices[ticker]
    
    # Step 2: Refresh put option data
    print(f"\n[INFO] Refreshing top {len(top_puts)} puts with {WORKERS} workers...")
    
    # Refresh puts concurrently
    refreshed_puts = []
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {}
        for idx, put in enumerate(top_puts):
            if idx > 0 and idx % WORKERS == 0:
                time.sleep(DELAY_BETWEEN_WORKERS)
            futures[executor.submit(refresh_put, put.copy())] = idx
        
        for future in as_completed(futures):
            result = future.result()
            refreshed_puts.append(result)
    
    # Filter out puts with negative time value
    refreshed_puts = [p for p in refreshed_puts if p.get("time_val", 0) >= 0]
    
    # Re-sort refreshed puts by rev_annual_rr_ratio
    refreshed_puts.sort(key=lambda x: x.get("rev_annual_rr_ratio") or float('-inf'), reverse=True)
    
    # Combine refreshed and remaining puts
    all_puts = refreshed_puts + remaining_puts
    
    # Re-rank all puts
    for rank, put in enumerate(all_puts, start=1):
        put["rank_order"] = rank
    
    # Save updated result_all.json with custom formatting (rank_order elevated above indented keys)
    def format_put_entry(put):
        lines = ['    {']
        lines.append(f'        "rank_order": {put["rank_order"]},')
        # Add remaining keys with 4 extra spaces of indentation
        keys = [k for k in put.keys() if k != "rank_order"]
        for i, key in enumerate(keys):
            val = put[key]
            if isinstance(val, str):
                val_str = f'"{val}"'
            elif val is None:
                val_str = 'null'
            else:
                val_str = json.dumps(val)
            comma = ',' if i < len(keys) - 1 else ''
            lines.append(f'            "{key}": {val_str}{comma}')
        lines.append('    }')
        return '\n'.join(lines)
    
    print(f"\n[INFO] Saving updated result_all.json...")
    with open("result_all.json", "w") as f:
        f.write('[\n')
        for i, put in enumerate(all_puts):
            f.write(format_put_entry(put))
            if i < len(all_puts) - 1:
                f.write(',')
            f.write('\n')
        f.write(']\n')
    
    # Calculate runtime
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"[DONE] Refreshed {len(unique_tickers)} stock prices")
    print(f"[DONE] Refreshed {len(refreshed_puts)} puts")
    print(f"[DONE] Total puts: {len(all_puts)}")
    print(f"[DONE] Runtime: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
