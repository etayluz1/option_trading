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
    Applies DTE (Days To Expiration) filtering from rules.json to ORATS data.
    """
    
    # 1. Load and parse the start_date and DTE rules from rules.json
    try:
        with open(rules_file_path, 'r') as f:
            rules = json.load(f)
            
            # Start Date
            start_date_str = rules["account_simulation"]["start_date"]
            start_date_obj = datetime.strptime(start_date_str, '%m/%d/%y').date()
            print(f"✅ Simulation start date loaded: {start_date_str} (Parsed as {start_date_obj})")
            
            # DTE Rules (New Extraction)
            min_dte_str = rules["entry_put_position"]["min_days_for_expiration"]
            max_dte_str = rules["entry_put_position"]["max_days_for_expiration"]
            
            # Convert DTE strings to integers
            # NOTE: These are the values you wanted: '1' and '500'
            MIN_DTE = int(min_dte_str)
            MAX_DTE = int(max_dte_str)
            print(f"✅ DTE Rules loaded: Min={MIN_DTE} days, Max={MAX_DTE} days.")
            
    except Exception as e:
        print(f"❌ Error loading/parsing rules.json or DTE values: {e}")
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

    # 3. Initialize Open Puts Tracker for ALL Tickers
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
    
    # Define the ORATS folder name once
    ORATS_FOLDER = "ORATS_json" 
    
    total_dates_processed = 0
    total_investable_entries_processed = 0
    
    for date_str in sorted_unique_dates:
        daily_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Check if the daily date is on or after the simulation start date
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
                # Check 1: Does the ticker have data for this date?
                if date_str in stock_history_dict[ticker]:
                    daily_data = stock_history_dict[ticker][date_str]
                    
                    # Check 2: Is the ticker investable on this date? (THE NEW FILTER)
                    if daily_data.get('investable') is True:
                        total_investable_entries_processed += 1
                        
                        # --- Trading Logic Placeholder ---
                        # Example Fictional Trade Logic: Sell 1 put on SPY every 3rd day
                        if ticker == TARGET_TICKER and daily_date_obj.day == 3:
                             open_puts_tracker[ticker] += 1
                        # -----------------------------------
                        
                        # --- REVISED: Extract and Filter days_interval values from ORATS data ---
                        # List to hold (expiration_date, days_interval) for all *filtered* chains
                        filtered_intervals = [] 
                        
                        if daily_orats_data and ticker in daily_orats_data:
                            ticker_orats_data = daily_orats_data[ticker]
                            
                            # Iterate over all expiration dates found for the current ticker
                            for expiration_date, exp_data in ticker_orats_data.items():
                                days_interval = exp_data.get('days_interval')
                                
                                # --- NEW FILTERING LOGIC ---
                                # Check if the value is an integer or a string that can be converted to an integer
                                if isinstance(days_interval, int) or (isinstance(days_interval, str) and days_interval.isdigit()):
                                    dte = int(days_interval) 
                                    
                                    # Check if DTE is within the allowed range
                                    if MIN_DTE <= dte <= MAX_DTE:
                                        # Only append the interval if it passes the DTE filter
                                        filtered_intervals.append((expiration_date, dte))
                                
                            
                        # Store the summary data in the dictionary
                        daily_investable_data[ticker] = {
                            'adj_close': daily_data.get('adj_close'),
                            'open_puts': open_puts_tracker[ticker],
                            'orats_intervals': filtered_intervals # Using the FILTERED list
                        }

            
            # Print the daily summary only if there were investable tickers
            if daily_investable_data:
                print(f"\n>>>> Date: {date_str} (Investable Tickers) <<<<")
                
                # Sort the tickers alphabetically
                sorted_tickers = sorted(daily_investable_data.keys())
                
                # Iterate over the sorted tickers to print
                for ticker in sorted_tickers:
                    data = daily_investable_data[ticker]
                    
                    # Format the list of intervals for printing
                    interval_str_parts = []
                    # Just print the interval value (DTE) from the FILTERED list
                    for _, interval in data['orats_intervals']: 
                        interval_str_parts.append(str(interval)) 
                        
                    interval_list_str = ", ".join(interval_str_parts) 
                    if not interval_list_str:
                        interval_list_str = f"None Found (DTE not in {MIN_DTE}-{MAX_DTE})" # Improved message
                        
                    print(
                        f"  | {ticker}: Adj.Close={data['adj_close']:<7} | Puts={data['open_puts']} | ORATS Intervals (DTE): [{interval_list_str}]"
                    )
            # --- END DAILY PROCESSING ---
    
    # 6. Print Final State of the Open Puts Tracker
    # ... (Step 6 continues here) ...

            
    print(f"\n--- Simulation Complete ---")
    print(f"Total unique trading days processed: {total_dates_processed}")
    print(f"Total investable daily ticker entries processed: {total_investable_entries_processed}")
    
    # 6. Print Final State of the Open Puts Tracker
    print("\n--- Final Open Puts Tally (ALL Tickers) ---")
    for ticker, count in open_puts_tracker.items():
        print(f"  {ticker:<5}: {count} open puts")

# Execute the main function (You would uncomment this to run the code)
load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)