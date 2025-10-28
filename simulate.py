import json
import os
from datetime import datetime

# --- Configuration ---
RULES_FILE_PATH = "rules.json"
JSON_FILE_PATH = "stock_history.json"
TARGET_TICKER = "SPY" # Retained for context, but the script processes ALL tickers.

def load_and_run_simulation(rules_file_path, json_file_path):
    """
    Loads rules and data, initializes the tracker, and iterates chronologically 
    over ALL daily entries for ALL tickers starting from the specified date.
    Only prints/processes tickers that are marked as "investable": true.
    Applies DTE (Days To Expiration) and pBidPx filtering from rules.json.
    """
    
    # 1. Load and parse the start_date, DTE, and BID PRICE rules from rules.json
    try:
        with open(rules_file_path, 'r') as f:
            rules = json.load(f)
            
            # Start Date
            start_date_str = rules["account_simulation"]["start_date"]
            start_date_obj = datetime.strptime(start_date_str, '%m/%d/%y').date()
            print(f"✅ Simulation start date loaded: {start_date_str} (Parsed as {start_date_obj})")
            
            # DTE Rules
            min_dte_str = rules["entry_put_position"]["min_days_for_expiration"]
            max_dte_str = rules["entry_put_position"]["max_days_for_expiration"]
            MIN_DTE = int(min_dte_str)
            MAX_DTE = int(max_dte_str)
            print(f"✅ DTE Rules loaded: Min={MIN_DTE} days, Max={MAX_DTE} days.")

            # Bid Price Rule
            min_bid_price_str = rules["entry_put_position"]["min_put_bid_price"]
            # Convert to float
            MIN_BID_PRICE = float(min_bid_price_str.replace('$', '').strip())
            print(f"✅ Bid Price Rule loaded: Min Bid Price > ${MIN_BID_PRICE:.2f}")

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
            
            # --- Load ORATS data for this specific date (once per day) ---
            orats_file_path = os.path.join(ORATS_FOLDER, f"{date_str}.json")
            daily_orats_data = None
            
            try:
                with open(orats_file_path, 'r') as f:
                    daily_orats_data = json.load(f)
            except FileNotFoundError:
                # Silently ignore if file doesn't exist for the day
                pass 
            except json.JSONDecodeError:
                print(f"  > ❌ ERROR: Could not decode JSON from '{orats_file_path}'. Skipping ORATS data.")
                pass
            
            # Inner loop: Check ALL tickers
            for ticker in all_tickers:
                # Check 1 & 2: Data exists for date AND ticker is investable
                if date_str in stock_history_dict[ticker] and stock_history_dict[ticker][date_str].get('investable') is True:
                    
                    total_investable_entries_processed += 1
                    daily_data = stock_history_dict[ticker][date_str]
                    
                    # --- Trading Logic Placeholder (Sell 1 put on SPY every 3rd day) ---
                    if ticker == TARGET_TICKER and daily_date_obj.day == 3:
                         open_puts_tracker[ticker] += 1
                    # -----------------------------------
                    
                    # --- DTE and pBidPx Filtering ---
                    # List to hold (expiration_date, DTE, list_of_filtered_options)
                    filtered_chains_summary = [] 
                    
                    if daily_orats_data and ticker in daily_orats_data:
                        ticker_orats_data = daily_orats_data[ticker]
                        
                        # Iterate over all expiration dates (chains)
                        for expiration_date, exp_data in ticker_orats_data.items():
                            days_interval = exp_data.get('days_interval')
                            options_array = exp_data.get('options', [])
                            
                            # --- DTE Filter Check ---
                            if isinstance(days_interval, int) or (isinstance(days_interval, str) and str(days_interval).isdigit()):
                                dte = int(days_interval) 
                                
                                if MIN_DTE <= dte <= MAX_DTE:
                                    
                                    # --- Bid Price Filter Loop ---
                                    filtered_options = []
                                    
                                    for option in options_array:
                                        # Assuming 'pBidPx' is the key for the put bid price
                                        pbidpx = float(option.get('pBidPx'))
                                        
                                        # Check if pBidPx exists and is greater than the minimum
                                        if isinstance(pbidpx, (int, float)) and pbidpx > MIN_BID_PRICE:
                                            # Option passes BOTH DTE and MIN BID criteria
                                            filtered_options.append(option)
                                            
                                    # Only record the chain if it has at least one option that passed the bid filter
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
                        # Store the detailed list of filtered chains
                        'filtered_options_summary': filtered_chains_summary 
                    }

            
            # Print the daily summary only if there were investable tickers
            if daily_investable_data:
                print(f"\n>>>> Date: {date_str} (Investable Tickers) <<<<")
                
                # Sort the tickers alphabetically
                sorted_tickers = sorted(daily_investable_data.keys())
                
                # Iterate over the sorted tickers to print
                for ticker in sorted_tickers:
                    data = daily_investable_data[ticker]
                    
                    # Prepare data for printing
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
                         # Print this message if the total count is 0
                         print(f"  |   > Details: None Found (Filtered out by DTE or Min Bid Price > ${MIN_BID_PRICE:.2f})")
                         
            # --- END DAILY PROCESSING ---
    
    # 6. Print Final State of the Open Puts Tracker
    print(f"\n--- Simulation Complete ---")
    print(f"Total unique trading days processed: {total_dates_processed}")
    print(f"Total investable daily ticker entries processed: {total_investable_entries_processed}")
    
    print("\n--- Final Open Puts Tally (ALL Tickers) ---")
    for ticker, count in open_puts_tracker.items():
        print(f"  {ticker:<5}: {count} open puts")

# Execute the main function
# To run this script, ensure you have 'rules.json', 'stock_history.json', and an 'ORATS_json' folder 
# populated with daily files in the same directory.
load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)