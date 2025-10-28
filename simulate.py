import json
import os
from datetime import datetime
from collections import defaultdict 

# --- Configuration ---
RULES_FILE_PATH = "rules.json"
JSON_FILE_PATH = "stock_history.json"
TARGET_TICKER = "SPY" # Retained for context, but the script processes ALL tickers.

def load_and_run_simulation(rules_file_path, json_file_path):
    """
    Loads rules and data, initializes the tracker, and iterates chronologically 
    over ALL daily entries for ALL tickers starting from the specified date.
    Only prints/processes tickers that are marked as "investable": true.
    """
    
    # 1. Load and parse the start_date from rules.json
    try:
        with open(rules_file_path, 'r') as f:
            rules = json.load(f)
            start_date_str = rules["account_simulation"]["start_date"]
            
            # Convert the "1/1/24" format string to a datetime.date object
            start_date_obj = datetime.strptime(start_date_str, '%m/%d/%y').date()
            print(f"✅ Simulation start date loaded: {start_date_str} (Parsed as {start_date_obj})")
            
    except Exception as e:
        print(f"❌ Error loading/parsing rules.json: {e}")
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
    
    total_dates_processed = 0
    total_investable_entries_processed = 0
    
    for date_str in sorted_unique_dates:
        daily_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Check if the daily date is on or after the simulation start date
        if daily_date_obj >= start_date_obj:
            
            # --- START DAILY PROCESSING ---
            total_dates_processed += 1
            # CHANGE: Use a dictionary to store data before sorting
            daily_investable_data = {} 
            
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
                        
                        # Store the summary data in the dictionary
                        daily_investable_data[ticker] = {
                            'adj_close': daily_data.get('adj_close'),
                            'open_puts': open_puts_tracker[ticker]
                        }

            
            # Print the daily summary only if there were investable tickers
            if daily_investable_data:
                print(f"\n>>>> Date: {date_str} (Investable Tickers) <<<<")
                
                # CHANGE: Sort the tickers alphabetically
                sorted_tickers = sorted(daily_investable_data.keys())
                
                # Iterate over the sorted tickers to print
                for ticker in sorted_tickers:
                    data = daily_investable_data[ticker]
                    print(
                        f"  | {ticker}: Adj.Close={data['adj_close']:<7} | Puts={data['open_puts']}"
                    )
            # --- END DAILY PROCESSING ---

            
    print(f"\n--- Simulation Complete ---")
    print(f"Total unique trading days processed: {total_dates_processed}")
    print(f"Total investable daily ticker entries processed: {total_investable_entries_processed}")
    
    # 6. Print Final State of the Open Puts Tracker
    print("\n--- Final Open Puts Tally (ALL Tickers) ---")
    for ticker, count in open_puts_tracker.items():
        print(f"  {ticker:<5}: {count} open puts")

# Execute the main function (You would uncomment this to run the code)
load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)