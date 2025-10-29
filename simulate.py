import json
import os
from datetime import datetime

# --- Configuration ---
RULES_FILE_PATH = "rules.json"
JSON_FILE_PATH = "stock_history.json"
TARGET_TICKER = "SPY" # Retained for context, but the script processes ALL tickers.

def safe_percentage_to_float(value):
    """Converts a percentage string (e.g., '-25%', '5.0%') to a decimal float (e.g., -0.25)."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            # Strip '%' and convert to float
            numeric_str = value.replace('%', '').strip()
            # Convert to decimal form (e.g., 5.0% -> 0.05)
            return float(numeric_str) / 100.0
        except ValueError:
            pass
    return None

def load_and_run_simulation(rules_file_path, json_file_path):
    """
    Loads rules and data, initializes the tracker, and iterates chronologically 
    over ALL daily entries for ALL tickers starting from the specified date.
    Applies DTE, Min Bid Price, Put Delta, Max Spread, and Strike Safety Margin filters.
    """
    
    # 1. Load and parse ALL rules from rules.json
    try:
        with open(rules_file_path, 'r') as f:
            rules = json.load(f)
            
            # Start Date
            start_date_str = rules["account_simulation"]["start_date"]
            start_date_obj = datetime.strptime(start_date_str, '%m/%d/%y').date()
            print(f"✅ Simulation start date loaded: {start_date_str} (Parsed as {start_date_obj})")
            
            # DTE Rules
            MIN_DTE = int(rules["entry_put_position"]["min_days_for_expiration"])
            MAX_DTE = int(rules["entry_put_position"]["max_days_for_expiration"])
            print(f"✅ DTE Rules loaded: Min={MIN_DTE} days, Max={MAX_DTE} days.")

            # Bid Price Rule
            min_bid_price_str = rules["entry_put_position"]["min_put_bid_price"]
            MIN_BID_PRICE = float(min_bid_price_str.replace('$', '').strip())
            print(f"✅ Bid Price Rule loaded: Min Bid Price > ${MIN_BID_PRICE:.2f}")
            
            # Put Delta Rules
            MIN_DELTA = safe_percentage_to_float(rules["entry_put_position"]["min_put_delta"])
            MAX_DELTA = safe_percentage_to_float(rules["entry_put_position"]["max_put_delta"])
            print(f"✅ Delta Rules loaded: {MIN_DELTA:.4f} <= Put Delta <= {MAX_DELTA:.4f}")
            
            # Max Bid-Ask Spread Rule
            MAX_SPREAD_DECIMAL = safe_percentage_to_float(rules["entry_put_position"]["max_ask_above_bid_pct"])
            print(f"✅ Spread Rule loaded: Max Ask above Bid = {MAX_SPREAD_DECIMAL * 100:.2f}%")
            
            # Strike Price Safety Margin Rule (NEW)
            MIN_AVG_ABOVE_STRIKE_PCT = safe_percentage_to_float(rules["entry_put_position"]["min_avg_above_strike"])
            # The required ratio for the SMA/Strike comparison
            REQUIRED_SMA_STRIKE_RATIO = 1.0 + MIN_AVG_ABOVE_STRIKE_PCT
            print(f"✅ Safety Rule loaded: SMA/Strike Ratio must be > {REQUIRED_SMA_STRIKE_RATIO:.4f}")


    except Exception as e:
        print(f"❌ Error loading/parsing rules.json values: {e}")
        return
    
    # 2. Load the main ticker data from stock_history.json
    try:
        with open(json_file_path, 'r') as f:
            stock_history_dict = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The data file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{json_file_path}'. Check file integrity.")
        return

    # 3. Initialize Open Puts Tracker 
    open_puts_tracker = {
        ticker: 0
        for ticker in stock_history_dict.keys()
    }
    
    all_tickers = list(open_puts_tracker.keys())
    print(f"✅ Open Puts Tracker initialized for {len(all_tickers)} tickers.")
    print("-" * 50)
    
    # 4. Determine the chronological order of all unique dates
    all_dates = set()
    for ticker_data in stock_history_dict.values():
        all_dates.update(ticker_data.keys())
        
    sorted_unique_dates = sorted(all_dates)
    
# 5. Iterate chronologically over the unique dates, processing investable tickers
    print(f"--- Starting Global Chronological Simulation from {start_date_obj} ---")
    
    ORATS_FOLDER = "ORATS_json" 
    
    total_dates_processed = 0
    total_investable_entries_processed = 0
    
    for date_str in sorted_unique_dates:
        daily_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        if daily_date_obj >= start_date_obj:
            
            # --- START DAILY PROCESSING ---
            total_dates_processed += 1
            daily_investable_data = {} 
            
            # Load ORATS data for this specific date
            orats_file_path = os.path.join(ORATS_FOLDER, f"{date_str}.json")
            daily_orats_data = None
            
            try:
                with open(orats_file_path, 'r') as f:
                    daily_orats_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            # Inner loop: Check ALL tickers
            for ticker in all_tickers:
                # Check 1 & 2: Data exists for date AND ticker is investable
                if date_str in stock_history_dict[ticker] and stock_history_dict[ticker][date_str].get('investable') is True:
                    
                    total_investable_entries_processed += 1
                    daily_data = stock_history_dict[ticker][date_str]
                    
                    # --- Stock Data Needed for Filtering ---
                    # We assume sma150_adj_close is calculated and stored in stock_history.json
                    sma150_adj_close = daily_data.get('sma150_adj_close')
                    
                    # Trading Logic Placeholder (Sell 1 put on SPY every 3rd day)
                    if ticker == TARGET_TICKER and daily_date_obj.day == 3:
                         open_puts_tracker[ticker] += 1
                    
                    # List to hold (expiration_date, DTE, list_of_filtered_options)
                    filtered_chains_summary = [] 
                    
                    if daily_orats_data and ticker in daily_orats_data:
                        ticker_orats_data = daily_orats_data[ticker]
                        
                        # Iterate over all expiration dates (chains)
                        for expiration_date, exp_data in ticker_orats_data.items():
                            days_interval = exp_data.get('days_interval')
                            options_array = exp_data.get('options', [])
                            
                            # --- 1. DTE Filter Check ---
                            if isinstance(days_interval, int) or (isinstance(days_interval, str) and str(days_interval).isdigit()):
                                dte = int(days_interval) 
                                
                                if MIN_DTE <= dte <= MAX_DTE:
                                    
                                    # --- 2. Bid Price, Delta, Spread, and Safety Margin Filter Loop ---
                                    filtered_options = []
                                    
                                    for option in options_array:
                                        
                                        # Parse necessary option data
                                        pbidpx_value = float(str(option.get('pBidPx', -1.0)).strip())
                                        paskpx_value = float(str(option.get('pAskPx', -1.0)).strip())
                                        put_delta_value = safe_percentage_to_float(option.get('putDelta'))
                                        
                                        # Strike price is often a float/number in ORATS data
                                        strike_value = float(option.get('strike', 0))
                                        
                                        # *** Filter Check A: Bid-Ask Spread ***
                                        passes_spread = False
                                        if pbidpx_value > 0 and paskpx_value > pbidpx_value:
                                            spread_pct = (paskpx_value - pbidpx_value) / pbidpx_value
                                            passes_spread = spread_pct <= MAX_SPREAD_DECIMAL

                                        # *** Filter Check B: Min Bid Price ***
                                        passes_bid = pbidpx_value > MIN_BID_PRICE
                                        
                                        # *** Filter Check C: Put Delta ***
                                        passes_delta = False
                                        if put_delta_value is not None:
                                            passes_delta = MIN_DELTA <= put_delta_value <= MAX_DELTA

                                        # *** Filter Check D: Strike Safety Margin (NEW) ***
                                        passes_safety_margin = False
                                        if sma150_adj_close is not None and strike_value > 0:
                                            # Check if sma150 is above the strike by the required ratio
                                            current_ratio = sma150_adj_close / strike_value
                                            passes_safety_margin = current_ratio > REQUIRED_SMA_STRIKE_RATIO
                                        
                                        # Option must pass ALL FOUR filters (A, B, C, D)
                                        if passes_bid and passes_delta and passes_spread and passes_safety_margin:
                                            filtered_options.append(option)
                                            
                                    # Only record the chain if it has at least one option that passed all filters
                                    if filtered_options:
                                        filtered_chains_summary.append({
                                            'expiration_date': expiration_date,
                                            'dte': dte,
                                            'filtered_options': filtered_options
                                        })

                    
                    # Store the summary data in the dictionary
                    daily_investable_data[ticker] = {
                        'adj_close': daily_data.get('adj_close'),
                        'open_puts': open_puts_tracker[ticker],
                        'filtered_options_summary': filtered_chains_summary 
                    }

            
            # Print the daily summary 
            if daily_investable_data:
                print(f"\n>>>> Date: {date_str} (Investable Tickers) <<<<")
                
                # Sort the tickers alphabetically
                sorted_tickers = sorted(daily_investable_data.keys())
                
                # Iterate over the sorted tickers to print
                for ticker in sorted_tickers:
                    data = daily_investable_data[ticker]
                    
                    summary_parts = []
                    total_filtered_options = 0
                    
                    for chain in data['filtered_options_summary']:
                        dte = chain['dte']
                        count = len(chain['filtered_options'])
                        total_filtered_options += count
                        summary_parts.append(f"{dte} days: {count} contracts")
                        
                    interval_list_str = "; ".join(summary_parts) 
                    
                    print(
                        f"  | {ticker}: Adj.Close={data['adj_close']:<7} | Puts={data['open_puts']} | Total Viable Options: {total_filtered_options}"
                    )
                    
                    # Print the detailed breakdown if contracts were found
                    if total_filtered_options > 0:
                         print(f"  |   > Details: {interval_list_str}")
                    else:
                         print(f"  |   > Details: None Found (Failed DTE, Bid, Delta, Spread, or Safety Margin filter)")
                         
            # --- END DAILY PROCESSING ---
    
    # 6. Print Final State of the Open Puts Tracker
    print(f"\n--- Simulation Complete ---")
    print(f"Total unique trading days processed: {total_dates_processed}")
    print(f"Total investable daily ticker entries processed: {total_investable_entries_processed}")
    
    print("\n--- Final Open Puts Tally (ALL Tickers) ---")
    for ticker, count in open_puts_tracker.items():
        print(f"  {ticker:<5}: {count} open puts")

# Execute the main function
load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)