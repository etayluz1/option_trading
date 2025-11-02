import json
import os
from datetime import datetime
import math
import sys
import time

# --- Logger Class ---
class Logger:
    """Redirects print statements to both the console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command, which is called by print().
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()

# --- Configuration ---
RULES_FILE_PATH = "rules.json"
JSON_FILE_PATH = "stock_history.json"
TARGET_TICKER = "SPY" # Retained for context, but the script processes ALL tickers.
DEBUG_VERBOSE = False # Set to True to see individual ticker details (Total Viable Options / Details by DTE)

# Commission Fee
COMMISSION_PER_CONTRACT = 0.67
FINAL_COMMISSION_PER_CONTRACT = COMMISSION_PER_CONTRACT # Commission for closing trades

# Maximum premium to collect per single trade entry
MAX_PREMIUM_PER_TRADE = 5000.00 

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

def calculate_risk_reward_ratio(strike, pBidPx):
    """
    Calculates the Risk/Reward Ratio: -(Risk / Reward) = -(Strike - pBidPx) / pBidPx.
    NOTE: Using the conventional negative ratio for selling puts.
    """
    if pBidPx > 0:
        # Risk is the maximum loss if stock goes to zero (Strike minus Premium received)
        risk = strike - pBidPx
        # Reward is the premium received
        reward = pBidPx
        # Return the negative ratio, as defined by MIN_RISK_REWARD_RATIO in rules.json
        return -(risk / reward)
    return None # Return None if calculation is invalid

def get_absolute_best_contract(ticker, filtered_summary):
    """
    Flattens the filtered_options_summary across all DTEs for a single ticker and 
    returns the contract with the highest calculated_rr_ratio.
    """
    all_viable_contracts = []
    
    # 1. Flatten all contracts into a single list
    for chain in filtered_summary:
        # Each option in the list already has the calculated_rr_ratio field
        for option in chain['filtered_options']:
            # Add DTE, expiration date, and ticker to the contract dictionary for easy reporting
            option['dte'] = chain['dte']
            option['expiration_date'] = chain['expiration_date']
            option['ticker'] = ticker
            all_viable_contracts.append(option)
            
    if not all_viable_contracts:
        return None
    
    # 2. Sort the entire list by calculated_rr_ratio in descending order (highest R/R first)
    all_viable_contracts.sort(key=lambda x: x.get('calculated_rr_ratio', -float('inf')), reverse=True)
    
    # 3. Return the first (best) contract
    return all_viable_contracts[0]

def print_daily_portfolio_summary(open_puts_tracker):
    """Prints a summary of all tickers with currently open put positions."""
    
    open_tickers = []
    total_open_positions = 0
    
    # 1. Collect and sort all open positions
    for ticker, count in open_puts_tracker.items():
        if count > 0:
            open_tickers.append((ticker, count))
            total_open_positions += count
            
    if not open_tickers:
        print("  (No open put positions.)")
        return
        
    open_tickers.sort(key=lambda x: x[0]) # Sort by ticker name
    
    # 2. Print the summary
    print(f"💼 **OPEN PORTFOLIO SUMMARY ({total_open_positions} Total Positions):**")
    
    summary_parts = []
    for ticker, count in open_tickers:
        summary_parts.append(f"{ticker}: {count}")
        
    print(f"  > {' | '.join(summary_parts)}")

def get_contract_exit_price(orats_data, ticker, expiration_date_str, strike):
    """
    Retrieves the conservative price to buy back (close) a short put position.
    Uses Ask Price, then Bid Price for conservative valuation.
    Returns: The exit price (cost to close) or None if contract is not found.
    """
    if not orats_data or ticker not in orats_data:
        return None

    ticker_data = orats_data[ticker]
    
    if expiration_date_str not in ticker_data:
        return None

    exp_data = ticker_data[expiration_date_str]
    options_array = exp_data.get('options', [])
    
    for option in options_array:
        option_strike = float(option.get('strike', 0))
        if abs(option_strike - strike) < 0.001: 
            try:
                pbidpx = float(str(option.get('pBidPx', 0.0)).strip())
                paskpx = float(str(option.get('pAskPx', 0.0)).strip())
            except ValueError:
                pbidpx = 0.0
                paskpx = 0.0
            
            # 1. Prioritize ASK PRICE (Most Conservative for closing a short position)
            if paskpx > 0:
                return paskpx
            # 2. Fallback to Bid Price (less conservative, but a value)
            elif pbidpx > 0:
                return pbidpx
            else:
                return 0.0 

    return None


def get_contract_bid_price(orats_data, ticker, expiration_date_str, strike):
    """
    Retrieves the current bid price for a given option contract from ORATS data.
    Returns the bid price (float) or None if not found or invalid.
    """
    if not orats_data or ticker not in orats_data:
        return None

    ticker_data = orats_data[ticker]
    if expiration_date_str not in ticker_data:
        return None

    exp_data = ticker_data[expiration_date_str]
    options_array = exp_data.get('options', [])

    for option in options_array:
        option_strike = float(option.get('strike', 0))
        if abs(option_strike - strike) < 0.001:
            try:
                pbidpx = float(str(option.get('pBidPx', 0.0)).strip())
            except (ValueError, TypeError):
                return None

            return pbidpx if pbidpx > 0 else None

    return None
    

def load_and_run_simulation(rules_file_path, json_file_path):
    """
    Loads rules and data, initializes the tracker, and iterates chronologically 
    over ALL daily entries for ALL tickers starting from the specified date.
    Implements exit logic, calculates daily P&L and account value, respects 
    portfolio limits, and sells the optimal quantity based on premium collected.
    """

    # --- LOGGING SETUP ---
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_file_number = 1
    while os.path.exists(os.path.join(LOG_DIR, f"{log_file_number}.log")):
        log_file_number += 1
    
    log_file_path = os.path.join(LOG_DIR, f"{log_file_number}.log")
    
    original_stdout = sys.stdout
    logger = Logger(log_file_path)
    sys.stdout = logger

    try:
        _run_simulation_logic(rules_file_path, json_file_path)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\nSimulation complete. Log saved to: {log_file_path}")

def _run_simulation_logic(rules_file_path, json_file_path):
    """Internal function containing the core simulation logic."""
    # Track wall-clock runtime of the whole simulation
    _sim_start_time = time.perf_counter()
    
    # 1. Load and parse ALL rules from rules.json
    try:
        with open(rules_file_path, 'r') as f:
            rules = json.load(f)
            
            # --- ACCOUNT LIMITS ---
            INITIAL_CASH = float(rules["account_simulation"]["initial_cash"].replace('$', '').replace(',', '').strip())
            MAX_PUTS_PER_ACCOUNT = int(rules["account_simulation"]["max_puts_per_account"])
            MAX_PUTS_PER_STOCK = int(rules["account_simulation"]["max_puts_per_stock"])
            
            # --- RISK MANAGEMENT RULE (Stock Price Stop Loss) ---
            STOCK_MAX_BELOW_AVG_PCT = abs(safe_percentage_to_float(rules["exit_put_position"]["stock_max_below_avg"]))
            
            # Start Date
            start_date_str = rules["account_simulation"]["start_date"]
            start_date_obj = datetime.strptime(start_date_str, '%m/%d/%y').date()
            
            # DTE Rules
            MIN_DTE = int(rules["entry_put_position"]["min_days_for_expiration"])
            MAX_DTE = int(rules["entry_put_position"]["max_days_for_expiration"])
            
            # Bid Price Rule
            MIN_BID_PRICE = float(rules["entry_put_position"]["min_put_bid_price"].replace('$', '').strip())
            
            # Put Delta Rules
            MIN_DELTA = safe_percentage_to_float(rules["entry_put_position"]["min_put_delta"])
            MAX_DELTA = safe_percentage_to_float(rules["entry_put_position"]["max_put_delta"])
            
            # Max Bid-Ask Spread Rule
            MAX_SPREAD_DECIMAL = safe_percentage_to_float(rules["entry_put_position"]["max_ask_above_bid_pct"])
            
            # Strike Price Safety Margin Rule
            MIN_AVG_ABOVE_STRIKE_PCT = safe_percentage_to_float(rules["entry_put_position"]["min_avg_above_strike"])
            REQUIRED_SMA_STRIKE_RATIO = 1.0 + MIN_AVG_ABOVE_STRIKE_PCT
            
            # Risk/Reward Ratio Rule
            MIN_RISK_REWARD_RATIO = float(rules["entry_put_position"]["min_risk_reward_ratio"])
            # Additional selection flags and thresholds
            USE_RR_SELECTOR = bool(rules['entry_put_position'].get('select_by_risk_reward_ratio', False))
            USE_ANNUAL_SELECTOR = bool(rules['entry_put_position'].get('select_by_annual_risk', False))
            USE_EXPECTED_SELECTOR = bool(rules['entry_put_position'].get('select_by_expected_profit', False))

            # Parse the additional numeric thresholds
            MIN_ANNUAL_RISK = safe_percentage_to_float(rules['entry_put_position'].get('min_annual_risk', None))
            MIN_EXPECTED_PROFIT = safe_percentage_to_float(rules['entry_put_position'].get('min_expected_profit', None))

            # Validate that only one selector is enabled at a time
            selectors_enabled = sum([1 if USE_RR_SELECTOR else 0,
                                     1 if USE_ANNUAL_SELECTOR else 0,
                                     1 if USE_EXPECTED_SELECTOR else 0])
            if selectors_enabled > 1:
                print("❌ Rule error: More than one entry-selection method is enabled in rules.json. Please enable only one of select_by_risk_reward_ratio, select_by_annual_risk, select_by_expected_profit.")
                return
            
            # Derive the per-trade premium budget from initial cash and account-level max positions
            # This replaces the hard-coded MAX_PREMIUM_PER_TRADE constant with a dynamic value.
            global MAX_PREMIUM_PER_TRADE
            try:
                if MAX_PUTS_PER_ACCOUNT > 0:
                    MAX_PREMIUM_PER_TRADE = float(INITIAL_CASH) / float(MAX_PUTS_PER_ACCOUNT)
                else:
                    # Fallback to existing constant if the rules are invalid
                    MAX_PREMIUM_PER_TRADE = MAX_PREMIUM_PER_TRADE
            except Exception:
                # In case of any unexpected parsing issues, keep the module-default
                pass

            # Define POSITION_STOP_LOSS_PCT first
            _pos_raw = rules.get('exit_put_position', {}).get('position_stop_loss_pct', "0%")
            POSITION_STOP_LOSS_PCT = abs(safe_percentage_to_float(_pos_raw)) if _pos_raw is not None else 0.0

            # Precompute Underlying Stock rules formatted strings (used in multiple tables)
            try:
                u_rules = rules.get('underlying_stock', {})
                u_min_rise = safe_percentage_to_float(u_rules.get('min_5_day_rise_pct'))
                u_min_above = safe_percentage_to_float(u_rules.get('min_above_avg_pct'))
                u_max_above = safe_percentage_to_float(u_rules.get('max_above_avg_pct'))
                u_min_slope = safe_percentage_to_float(u_rules.get('min_avg_up_slope_pct'))
                try:
                    u_min_price = float(str(u_rules.get('min_stock_price', '')).replace('$', '').replace(',', '').strip())
                except Exception:
                    u_min_price = None

                # Consistent formatting widths across summaries
                min_rise_str = f"{u_min_rise*100:>13.1f}%" if u_min_rise is not None else f"{'N/A':>13}"
                min_above_str = f"{u_min_above*100:>13.1f}%" if u_min_above is not None else f"{'N/A':>13}"
                max_above_str = f"{u_max_above*100:>13.1f}%" if u_max_above is not None else f"{'N/A':>13}"
                min_slope_str = f"{u_min_slope*100:>13.1f}%" if u_min_slope is not None else f"{'N/A':>13}"
                min_price_str = f"$ {u_min_price:>12.2f}" if u_min_price is not None else f"{'N/A':>14}"
            except Exception:
                # Provide fallbacks if rules are missing or malformed
                min_rise_str = f"{'N/A':>13}"
                min_above_str = f"{'N/A':>13}"
                max_above_str = f"{'N/A':>13}"
                min_slope_str = f"{'N/A':>13}"
                min_price_str = f"{'N/A':>14}"

            # Print rules in formatted tables
            print("\n=== TRADING RULES SUMMARY ===\n")
            
            # 1. Account Simulation Rules
            print("📊 Account Simulation Rules")
            print("|--------------------|----------------|")
            print("| Parameter          | Value          |")
            print("|--------------------|----------------|")
            print(f"| Start Date         | {start_date_str:<14} |")
            print(f"| Initial Cash       | ${float(rules['account_simulation']['initial_cash']):>13,.2f} |")
            print(f"| Max Puts/Account   | {MAX_PUTS_PER_ACCOUNT:>14} |")
            print(f"| Max Puts/Stock     | {MAX_PUTS_PER_STOCK:>14} |")
            print("|--------------------|----------------|")
            print()
            
            # 2.b Underlying Stock Rules (precomputed values)
            print("🧩 Underlying Stock Rules")
            print("|----------------------------|----------------|")
            print("| Parameter                  | Value          |")
            print("|----------------------------|----------------|")
            print(f"| Min 5-Day Rise             | {min_rise_str} |")
            print(f"| Min Above Avg              | {min_above_str} |")
            print(f"| Max Above Avg              | {max_above_str} |")
            print(f"| Min 10-Day Avg Slope       | {min_slope_str} |")
            print(f"| Min Stock Price            | {min_price_str} |")
            print("|----------------------------|----------------|")
            print()

            # 3. Entry Put Position Rules
            print("📈 Entry Put Position Rules")
            print("|------------------------|----------------|")
            print("| Parameter              | Value          |")
            print("|------------------------|----------------|")
            print(f"| Min DTE                | {MIN_DTE:>14} |")
            print(f"| Max DTE                | {MAX_DTE:>14} |")
            print(f"| Min Put Bid Price      | $ {MIN_BID_PRICE:>12.2f} |")
            print(f"| Min Put Delta          | {MIN_DELTA*100:>13.1f}% |")
            print(f"| Max Put Delta          | {MAX_DELTA*100:>13.1f}% |")
            print(f"| Max Bid-Ask Spread     | {MAX_SPREAD_DECIMAL*100:>13.1f}% |")
            print(f"| Min Avg Above Strike   | {MIN_AVG_ABOVE_STRIKE_PCT*100:>13.1f}% |")
            print(f"| Min Risk/Reward Ratio  | {MIN_RISK_REWARD_RATIO:>14.1f} |")
            print(f"| Min Annual Risk        | {safe_percentage_to_float(rules['entry_put_position']['min_annual_risk'])*100:>13.1f}% |")
            print(f"| Min Expected Profit    | {safe_percentage_to_float(rules['entry_put_position']['min_expected_profit'])*100:>13.1f}% |")
            print(f"| Use Risk/Reward Ratio  | {('Yes' if rules['entry_put_position']['select_by_risk_reward_ratio'] else 'No'):>14} |")
            print(f"| Use Annual Risk        | {('Yes' if rules['entry_put_position']['select_by_annual_risk'] else 'No'):>14} |")
            print(f"| Use Expected Profit    | {('Yes' if rules['entry_put_position']['select_by_expected_profit'] else 'No'):>14} |")
            print("|------------------------|----------------|")
            print()

            # 4. Exit Put Position Rules
            print("📉 Exit Put Position Rules")
            print("|--------------------|--------------|")
            print("| Parameter          | Value        |")
            print("|--------------------|--------------|")
            print(f"| Position Stop Loss | {POSITION_STOP_LOSS_PCT*100:>11.1f}% |")
            print(f"| Stock Below SMA150 | {STOCK_MAX_BELOW_AVG_PCT*100:>11.1f}% |")
            print("|--------------------|--------------|")
            print()

            # 4. Trading Costs and Limits
            print("💰 Trading Parameters")
            print("|--------------------|--------------|")
            print("| Parameter          | Value        |")
            print("|--------------------|--------------|")
            print(f"| Commission/Contract| ${COMMISSION_PER_CONTRACT:>11.2f} |")
            print(f"| Max Premium/Trade  | ${MAX_PREMIUM_PER_TRADE:>11.2f} |")
            print("|--------------------|--------------|")
            print()
            
            # Stop here to view the tables
            # quit()
            # Position stop-loss rule: threshold is compared against daily option BID vs the entry BID
            # The rule is defined in `exit_put_position.position_stop_loss_pct` in rules.json
            _pos_raw = rules.get('exit_put_position', {}).get('position_stop_loss_pct', "0%")
            POSITION_STOP_LOSS_PCT = abs(safe_percentage_to_float(_pos_raw)) if _pos_raw is not None else 0.0
            print(f"✅ Position Stop Loss Threshold (daily bid loss vs entry bid): {POSITION_STOP_LOSS_PCT * 100:.2f}% (sourced from exit_put_position.position_stop_loss_pct)")


    except Exception as e:
        print(f"❌ Error loading/parsing rules.json values: {e}")
        return
    
    # 2. Load the main ticker data from stock_history.json
    try:
        with open(json_file_path, 'r') as f:
            stock_history_dict = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The data file '{json_file_path}' was was not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could could decode JSON from '{json_file_path}'. Check file integrity.")
        return

    # 3. Initialize Trackers 
    # Tracks count of open puts per ticker
    open_puts_tracker = {
        ticker: 0
        for ticker in stock_history_dict.keys()
    }
    # Tracks detailed trade entries (Strike, Expiration, etc.). Crucial for checking duplicates.
    open_trades_log = [] 
    # Log for all closed positions
    closed_trades_log = []
    
    # --- Financial Trackers ---
    cash_balance = INITIAL_CASH # Tracks cash directly
    cumulative_realized_pnl = 0.0 # Tracks profit from closed trades
    
    # --- Performance Tracking ---
    sim_start_date = None
    sim_end_date = None
    spy_start_price = None
    spy_end_price = None
    
    # NEW TRACKER: Total number of distinct trade entry events
    total_entry_events = 0
    # NEW TRACKER: Total number of distinct trade exit events
    total_exit_events = 0
    # Total contracts entered (sum of Qty) - kept for internal consistency check
    total_contracts_opened_qty = 0 

    # NEW: SPY tracking for monthly/yearly
    monthly_spy_prices = {} 
    
    # NEW: Monthly P&L Log (Tuple: (Realized PNL for month, EOD Total Value, SPY Close Price))
    monthly_pnl_log = {} 
    
    # NEW: Exit Counters (Used for contract quantity breakdown)
    stop_loss_count = 0
    expired_otm_count = 0
    expired_itm_count = 0

    # NEW: Gain Trackers for Attribution Table
    stop_loss_gain = 0.0
    stop_loss_premium_collected = 0.0
    expired_otm_gain = 0.0
    expired_otm_premium_collected = 0.0
    expired_itm_gain = 0.0
    expired_itm_premium_collected = 0.0
    liquidation_gain = 0.0
    liquidation_premium_collected = 0.0

    # NEW: Track peak NAV to compute current drawdown
    peak_account_value = INITIAL_CASH
    # NEW: Track worst drawdown percentage observed across all simulated dates (negative number, e.g., -12.34)
    worst_drawdown_pct = 0.0

    all_tickers = list(open_puts_tracker.keys())
    print(f"✅ Trackers initialized for {len(all_tickers)} tickers.")
    print("-" * 50)
    
    # Helper: determine if a given day's data meets the 'investable' criteria
    def compute_investable_flag(daily_data, rules):
        """Return True if the day's metrics satisfy the underlying_stock entry filters.

        This mirrors the logic used in get_stock_history.py but operates on the
        already-loaded `daily_data` dict (which contains strings like "5.000%" for
        many metrics). It returns a boolean.
        """
        try:
            # Local helper to convert percent-strings like '5.000%' -> 5.0
            def pct_str_to_percent(value):
                if value is None:
                    return None
                v = safe_percentage_to_float(value)
                return (v * 100.0) if v is not None else None

            # Extract day's metrics (strings with % or numeric adj_close)
            day_rise = pct_str_to_percent(daily_data.get('5_day_rise'))
            adj_above_pct = pct_str_to_percent(daily_data.get('adj_price_above_avg_pct'))
            sma_slope = pct_str_to_percent(daily_data.get('10_day_avg_slope'))
            adj_close = daily_data.get('adj_close')

            # Parse rules and convert to comparable units (percent values as raw numbers)
            min_5_day_rise_pct = safe_percentage_to_float(rules['underlying_stock']['min_5_day_rise_pct'])
            min_5_day_rise_pct = min_5_day_rise_pct * 100.0 if min_5_day_rise_pct is not None else None

            min_above_avg_pct = safe_percentage_to_float(rules['underlying_stock']['min_above_avg_pct'])
            max_above_avg_pct = safe_percentage_to_float(rules['underlying_stock']['max_above_avg_pct'])
            if min_above_avg_pct is not None:
                min_above_avg_pct *= 100.0
            if max_above_avg_pct is not None:
                max_above_avg_pct *= 100.0

            min_avg_up_slope_pct = safe_percentage_to_float(rules['underlying_stock']['min_avg_up_slope_pct'])
            min_avg_up_slope_pct = min_avg_up_slope_pct * 100.0 if min_avg_up_slope_pct is not None else None

            # Price rule (strip $)
            min_stock_price_rule = None
            try:
                min_stock_price_rule = float(str(rules['underlying_stock'].get('min_stock_price', '')).replace('$', '').strip())
            except Exception:
                min_stock_price_rule = None

            # If any essential metric is missing, conservatively mark as not investable
            if day_rise is None or adj_above_pct is None or sma_slope is None or adj_close is None:
                return False

            # Apply the same conditions used in get_stock_history.py
            cond_rise = day_rise > (min_5_day_rise_pct if min_5_day_rise_pct is not None else -1e9)
            cond_above_avg = True
            if min_above_avg_pct is not None and max_above_avg_pct is not None:
                cond_above_avg = (adj_above_pct >= min_above_avg_pct) and (adj_above_pct <= max_above_avg_pct)

            cond_slope = sma_slope > (min_avg_up_slope_pct if min_avg_up_slope_pct is not None else -1e9)
            cond_price = True
            if min_stock_price_rule is not None:
                try:
                    cond_price = float(adj_close) > float(min_stock_price_rule)
                except Exception:
                    cond_price = False

            return cond_rise and cond_above_avg and cond_slope and cond_price
        except Exception:
            return False
    
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
    
    # Set for quick checking of active positions (Ticker, Strike, Expiration)
    active_position_keys = set()
    
    # Variable to hold last day's ORATS data for final liquidation (if needed)
    last_daily_orats_data = None 
    
    for date_str in sorted_unique_dates:
        daily_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        if daily_date_obj >= start_date_obj:
            
            # --- START DAILY PROCESSING ---
            # Capture simulation dates and SPY prices
            print()              
            print(daily_date_obj)
            spy_current_price = None
            if 'SPY' in stock_history_dict and date_str in stock_history_dict['SPY']:
                spy_current_price = stock_history_dict['SPY'][date_str].get('adj_close')
                
            if sim_start_date is None:
                sim_start_date = daily_date_obj
                spy_start_price = spy_current_price
            
            sim_end_date = daily_date_obj # Update end date every successful day
            spy_end_price = spy_current_price # Update end SPY price every day

            # Store SPY price for the current month/year
            month_key = (daily_date_obj.year, daily_date_obj.month)
            if spy_current_price is not None:
                 monthly_spy_prices[month_key] = spy_current_price
            
            total_dates_processed += 1
            daily_investable_data = {} 
            daily_trade_candidates = [] 
            
            # Load ORATS data for this specific date
            orats_file_path = os.path.join(ORATS_FOLDER, f"{date_str}.json")
            daily_orats_data = None
            try:
                with open(orats_file_path, 'r') as f:
                    daily_orats_data = json.load(f)
                    last_daily_orats_data = daily_orats_data # Store the latest successful load
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            current_account_put_positions = sum(open_puts_tracker.values())
            daily_pnl = 0.0 # Realized P&L from closed trades today
            
            # --- Liability Trackers (Re-initialized daily for fresh MTM calculation) ---
            unrealized_pnl = 0.0 
            total_put_liability = 0.0 
            total_open_premium_collected = 0.0 
            
            # FIX 1: Initialize total_account_value and account_full_today at the start of the loop
            total_account_value = cash_balance # Placeholder; actual MTM calculated later
            account_full_today = False 

            # ----------------------------------------------------
            # --- Position Management/Exit Logic (Stop-Loss & Expiration) ---
            # ----------------------------------------------------
            positions_to_remove = []
            
            for i, trade in enumerate(open_trades_log):
                ticker = trade['ticker']
                
                # --- Get Stock Data and Option Exit Price for Today ---
                current_stock_data = stock_history_dict.get(ticker, {}).get(date_str, {})
                current_adj_close = current_stock_data.get('adj_close')
                sma150_adj_close = current_stock_data.get('sma150_adj_close')
                
                # Get conservative option price (Ask price) for stop-loss closure
                current_ask_price = get_contract_exit_price(
                    daily_orats_data, 
                    trade['ticker'], 
                    trade['expiration_date'], 
                    trade['strike']
                )

                # 1. STOP-LOSS CHECK (If data is available)
                stop_loss_triggered = False
                if current_adj_close is not None and sma150_adj_close is not None:
                    
                    # Calculate the threshold: SMA150 * (1 - Max Drop %)
                    threshold = sma150_adj_close * (1.0 - STOCK_MAX_BELOW_AVG_PCT)
                    
                    if current_adj_close < threshold:
                        stop_loss_triggered = True

                # 1.b POSITION-LEVEL STOP-LOSS (based on option BID movement vs entry BID)
                # If the option's daily BID moves above the entry BID by the configured percentage,
                # treat it as a stop-loss and exit on the same day.
                position_stop_loss_triggered = False
                try:
                    current_bid_price = get_contract_bid_price(
                        daily_orats_data,
                        trade['ticker'],
                        trade['expiration_date'],
                        trade['strike']
                    )
                except Exception:
                    current_bid_price = None

                entry_bid_price = trade.get('premium_received')
                if current_bid_price is not None and entry_bid_price is not None and entry_bid_price > 0:
                    # Loss ratio relative to entry bid (positive when current bid > entry bid)
                    loss_ratio = (current_bid_price - entry_bid_price) / entry_bid_price
                    if loss_ratio > 0 and loss_ratio >= POSITION_STOP_LOSS_PCT:
                        position_stop_loss_triggered = True
                        # mark as a stop loss so existing exit flow is used
                        stop_loss_triggered = True
                        # ensure we have a price to use for closing; prefer ASK, then BID
                        if current_ask_price is None:
                            # try to fetch an ask price specifically; get_contract_exit_price prioritizes ask
                            fallback_ask = get_contract_exit_price(
                                daily_orats_data,
                                trade['ticker'],
                                trade['expiration_date'],
                                trade['strike']
                            )
                            current_ask_price = fallback_ask if fallback_ask is not None else current_bid_price
                        
                # 2. EXPIRATION CHECK
                try:
                    exp_date_obj = datetime.strptime(trade['expiration_date'], '%m/%d/%Y').date() 
                except ValueError:
                    try:
                        exp_date_obj = datetime.strptime(trade['expiration_date'], '%m/%d/%y').date() 
                    except ValueError:
                        from datetime import timedelta
                        exp_date_obj = daily_date_obj + timedelta(days=3650) 
                
                expired_triggered = (exp_date_obj <= daily_date_obj)


                if stop_loss_triggered or expired_triggered:
                    
                    qty = trade['quantity'] 
                    
                    # Store exit details temporarily
                    exit_details = {
                        'DayOut': daily_date_obj.strftime('%Y-%m-%d'),
                        'PriceOut': None, # Option Price at Exit (Bid/Ask/0)
                        'QtyOut': qty,
                        'AmountOut': 0.0, # Gross cost to close (premium/payout)
                        'ReasonWhyClosed': None,
                        'Gain$': 0.0,
                        'Gain%': 0.0,
                    }
                    
                    # --- Calculate Exit P&L ---
                    exit_commission = qty * FINAL_COMMISSION_PER_CONTRACT
                    premium_collected_gross = trade['premium_received'] * qty * 100.0
                    
                    if expired_triggered:
                        # --- CRITICAL FIX FOR OTM/ITM ASSIGNMENT ---
                        
                        if current_adj_close is None:
                            continue 
                            
                        is_itm = current_adj_close < trade['strike']
                        
                        if is_itm:
                            # ITM/ASSIGNMENT SCENARIO (Loss)
                            assignment_loss_gross = (trade['strike'] - current_adj_close) * qty * 100.0 + exit_commission 
                            net_profit = premium_collected_gross - assignment_loss_gross - exit_commission
                            
                            expired_itm_count += qty
                            expired_itm_gain += net_profit
                            expired_itm_premium_collected += premium_collected_gross
                            
                            exit_details['PriceOut'] = current_adj_close # Stock price at assignment
                            exit_details['AmountOut'] = assignment_loss_gross # Gross assignment loss
                            exit_details['ReasonWhyClosed'] = "Expiration (ITM/Assigned)"
                            cost_to_close_gross = assignment_loss_gross
                            
                        else:
                            # OTM/MAX PROFIT SCENARIO
                            cost_to_close_gross = 0.0 
                            exit_commission = 0.0
                            net_profit = premium_collected_gross - exit_commission
                            
                            expired_otm_count += qty
                            expired_otm_gain += net_profit
                            expired_otm_premium_collected += premium_collected_gross
                            
                            exit_details['PriceOut'] = 0.0 # Option price at expiration
                            exit_details['AmountOut'] = 0.0 # Gross cost to close
                            exit_details['ReasonWhyClosed'] = "Expiration (OTM/Max Profit)"
                            
                        # --- END CRITICAL FIX ---
                        
                    elif stop_loss_triggered and current_ask_price is not None:
                        # STOP LOSS SCENARIO (stock-level stop OR position-level stop)
                        cost_to_close_gross = current_ask_price * qty * 100.0 + exit_commission
                        net_profit = premium_collected_gross - cost_to_close_gross - exit_commission

                        # Determine specific stop-loss reason (position-level vs stock-level)
                        if 'position_stop_loss_triggered' in locals() and position_stop_loss_triggered:
                            reason = "POSITION STOP LOSS (Bid moved above entry threshold)"
                        else:
                            reason = "STOP LOSS (Stock Below SMA150)"

                        stop_loss_count += qty
                        stop_loss_gain += net_profit
                        stop_loss_premium_collected += premium_collected_gross

                        exit_details['PriceOut'] = current_ask_price # Option Ask Price at Exit
                        exit_details['AmountOut'] = cost_to_close_gross # Gross cost to close
                        exit_details['ReasonWhyClosed'] = reason
                    else:
                        # Cannot determine exit price for stop-loss, skip for now 
                        continue 
                    
                    # FINAL EXIT EVENT COUNT: Increment by 1 for every unique trade closed
                    total_exit_events += 1
                    
                    # --- CASH FLOW TRANSPARENCY & NET PROFIT APPLICATION ---
                    cash_before_event = cash_balance # Capture Cash before this realized event
                    
                    # The cost of the buy-back/payout (market price)
                    market_cost_to_close = exit_details['AmountOut'] 
                    
                    # STEP 1: DEBIT - Pay the Cost to Close (Buy to Cover/Payout)
                    cash_balance -= market_cost_to_close # includes payout + commission
                    
                    # STEP 2: DEBIT - Pay Commission
                    # cash_balance -= exit_commission # Already included in market_cost_to_close above
                    
                    # STEP 3: CREDIT - Restore the Collected Premium. 
                    # FIX: This step IS required to realize the NET profit (Premium collected - Costs)
                    # If we don't add the premium back, the cash balance will be Cash_Old - Cost_Close - Commission, 
                    # which is NOT the net P&L realized. The current simulation design uses the full premium 
                    # as part of the cash base, so it must be added back to complete the realization.
                    # Yuda: I don't like this bug cash_balance += premium_collected_gross
                    
                    # Final P&L calculation
                    daily_pnl += net_profit
                    cumulative_realized_pnl += net_profit

                    
                    # Calculate percentage gain relative to max risk (Premium / Max Loss)
                    premium_collected_per_contract = trade['premium_received'] * 100.0
                    max_risk_per_contract = (trade['strike'] * 100.0) - premium_collected_per_contract
                    
                    if max_risk_per_contract > 0:
                        position_gain_percent = (net_profit / (max_risk_per_contract * qty)) * 100.0
                    else:
                        position_gain_percent = 0.0

                    exit_details['Gain$'] = net_profit
                    exit_details['Gain%'] = position_gain_percent

                    # Log the exit
                    print(f"🔥 **EXIT:** {exit_details['ReasonWhyClosed']}: {trade['ticker']} (Strike ${trade['strike']:.2f}, Qty {qty}). Net Profit: ${net_profit:,.2f}")
                    
                    # --- NEW LOGGING FOR CASH FLOW TRANSPARENCY ---
                    # FIX: Displaying the three components that net out to the P&L, proving the net change is correct.
                    print(f"  | **Cash Balance Before Event:** ${cash_before_event:,.2f}")
                    print(f"  | - Cash Outflow (Buy to Cover @ Ask/Payout): -${market_cost_to_close:,.2f}")
                    print(f"  | - Commission: -${exit_commission:,.2f}")
                    print(f"  | + Premium Collected (Realized Component): +${premium_collected_gross:,.2f}")
                    print(f"  | **Final Cash Balance After Event:** ${cash_balance:,.2f} (Net Change: ${net_profit:,.2f})")
                    # --- END NEW LOGGING ---

                    # --- CAPTURE COMPLETE TRADE LOG AND REMOVE ---
                    
                    # Consolidate entry details (using separate keys for clarity in the final table)
                    trade_to_log = {
                        'Ticker': trade['ticker'],
                        'Strike': trade['strike'],
                        'ExpDate': trade['expiration_date'],
                        'DayIn': trade['entry_date'],
                        'PriceIn': trade['premium_received'], # Option Bid Price in
                        'Qty': qty,
                        'AmountIn': premium_collected_gross, # Gross premium collected
                        **exit_details # Merge in all exit details
                    }
                    closed_trades_log.append(trade_to_log)
                    
                    # Update trackers
                    positions_to_remove.append(i)
                    open_puts_tracker[trade['ticker']] -= 1 
                    active_position_keys.remove(trade['unique_key'])
            
            # --- Monthly and Yearly P&L Aggregation (Start of Day P&L aggregation) ---
            month_key = (daily_date_obj.year, daily_date_obj.month)
            
            # Monthly PNL is now tracked as (Realized PNL, MTM Value, SPY Close)
            if month_key not in monthly_pnl_log:
                monthly_pnl_log[month_key] = (daily_pnl, 0.0, 0.0) 
            else:
                current_pnl, _, _ = monthly_pnl_log[month_key]
                monthly_pnl_log[month_key] = (current_pnl + daily_pnl, 0.0, 0.0) 
            
            # Update cumulative P&L with today's realized profit (Net of exit commissions)
            cumulative_realized_pnl += daily_pnl
            
            # Remove closed positions from the log, iterating backwards
            for index in sorted(positions_to_remove, reverse=True):
                open_trades_log.pop(index)
            
            # Recalculate current_account_put_positions after exits
            current_account_put_positions = sum(open_puts_tracker.values())

            # --- DYNAMIC RISK SIZING: Recalculate Max Premium Per Trade for THIS DAY ---
            # Compute conservative NAV before new entries: Cash minus cost to close current open puts
            total_put_liability = 0.0
            for ot in open_trades_log:
                price_for_ot = get_contract_exit_price(
                    daily_orats_data,
                    ot['ticker'],
                    ot['expiration_date'],
                    ot['strike']
                )
                if price_for_ot is not None:
                    total_put_liability += price_for_ot * ot['quantity'] * 100.0

            # NAV (Total Account Value) used for sizing = cash_balance - total_put_liability
            total_account_value = cash_balance - total_put_liability
            if total_account_value < 0:
                # Prevent negative sizing
                total_account_value = 0.0

            # Daily max premium per trade = NAV divided by max allowed positions
            if MAX_PUTS_PER_ACCOUNT > 0:
                max_premium_per_trade_today = total_account_value / float(MAX_PUTS_PER_ACCOUNT)
            else:
                max_premium_per_trade_today = MAX_PREMIUM_PER_TRADE

            # Ensure a sensible floor (avoid extremely small or zero budgets)
            if max_premium_per_trade_today <= 0:
                max_premium_per_trade_today = min(MAX_PREMIUM_PER_TRADE, 100.0)

            print(f"📈 Max Premium per Trade (today, NAV/{MAX_PUTS_PER_ACCOUNT}): ${max_premium_per_trade_today:,.2f}")


            # ----------------------------------------------
            # --- Market Scan and Trade Entry ---
            # ----------------------------------------------
            
            # Check if we should skip the market scan due to global limits
            if current_account_put_positions >= MAX_PUTS_PER_ACCOUNT and not account_full_today:               
                print(f"🛑 **ACCOUNT FULL (Global Limit):** {current_account_put_positions}/{MAX_PUTS_PER_ACCOUNT} contracts. Skipping scan for new trades.")
                account_full_today = True
            
            # Print the daily header and limits info for scan days                                   

            if not account_full_today:
                
                # Inner loop: Check ALL tickers for viable contracts
                for ticker in all_tickers:
                    # Check 1: Data exists for date
                    if date_str in stock_history_dict[ticker]:
                        # Compute investable dynamically from the day's metrics and rules
                        daily_data = stock_history_dict[ticker][date_str]
                        is_investable = compute_investable_flag(daily_data, rules)
                        # Set the flag so downstream code (and prints) can still observe it
                        daily_data['investable'] = is_investable
                        if not is_investable:
                            continue # Skip non-investable tickers
                        
                        # Optimization: Skip scanning this ticker if its limit is reached
                        if open_puts_tracker[ticker] >= MAX_PUTS_PER_STOCK:
                            continue
                            
                        total_investable_entries_processed += 1
                        daily_data = stock_history_dict[ticker][date_str]
                        
                        # --- Stock Data Needed for Filtering ---
                        sma150_adj_close = daily_data.get('sma150_adj_close')
                        current_adj_close = daily_data.get('adj_close')
                        
                        
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
                                        
                                        # --- 2. All Contract Filters Loop ---
                                        filtered_options = []
                                        
                                        for option in options_array:
                                            
                                            # Parse necessary option data
                                            pbidpx_value = float(str(option.get('pBidPx', -1.0)).strip())
                                            paskpx_value = float(str(option.get('pAskPx', -1.0)).strip())
                                            put_delta_value = safe_percentage_to_float(option.get('putDelta'))
                                            strike_value = float(option.get('strike', 0))
                                            
                                            # --- Check 1: Min Bid Price ---
                                            passes_bid = pbidpx_value > MIN_BID_PRICE
                                            if not passes_bid: continue # Fail fast
                                            
                                            # --- Check 2: Put Delta ---
                                            passes_delta = False
                                            if put_delta_value is not None:
                                                passes_delta = MIN_DELTA <= put_delta_value <= MAX_DELTA
                                            if not passes_delta: continue # Fail fast

                                            # --- Check 3: Bid-Ask Spread ---
                                            passes_spread = False
                                            if pbidpx_value > 0 and paskpx_value > pbidpx_value:
                                                spread_pct = (paskpx_value - pbidpx_value) / pbidpx_value
                                                passes_spread = spread_pct <= MAX_SPREAD_DECIMAL
                                            if not passes_spread: continue # Fail fast

                                            # --- Check 4: Strike Safety Margin ---
                                            passes_safety_margin = False
                                            if sma150_adj_close is not None and strike_value > 0:
                                                current_ratio = sma150_adj_close / strike_value
                                                passes_safety_margin = current_ratio > REQUIRED_SMA_STRIKE_RATIO
                                            if not passes_safety_margin: continue # Fail fast

                                            # --- Check 5: Risk/Reward Ratio ---
                                            passes_metric = False
                                            risk_reward_ratio = None
                                            annual_rr = None
                                            expected_profit = None
                                            
                                            if pbidpx_value > 0 and strike_value > pbidpx_value:
                                                # Base R/R ratio
                                                risk_reward_ratio = calculate_risk_reward_ratio(strike_value, pbidpx_value)
                                                # Compute annualized R/R (if DTE available)
                                                try:
                                                    if risk_reward_ratio is not None and isinstance(dte, int) and dte > 0:
                                                        annual_rr = risk_reward_ratio * (365.0 / float(dte))
                                                except Exception:
                                                    annual_rr = None

                                                # Compute expected profit metric when Delta is available
                                                try:
                                                    if put_delta_value is not None:
                                                        # put_delta_value is a decimal (e.g., -0.2)
                                                        expected_profit = (pbidpx_value * (1.0 + put_delta_value) + (strike_value - pbidpx_value) * put_delta_value) / pbidpx_value
                                                except Exception:
                                                    expected_profit = None

                                                # Choose which metric to apply as the filter based on rules
                                                if USE_ANNUAL_SELECTOR:
                                                    if annual_rr is not None and MIN_ANNUAL_RISK is not None:
                                                        passes_metric = annual_rr > MIN_ANNUAL_RISK
                                                elif USE_EXPECTED_SELECTOR:
                                                    if expected_profit is not None and MIN_EXPECTED_PROFIT is not None:
                                                        passes_metric = expected_profit > MIN_EXPECTED_PROFIT
                                                else:
                                                    # Default or USE_RR_SELECTOR
                                                    if risk_reward_ratio is not None:
                                                        passes_metric = risk_reward_ratio > MIN_RISK_REWARD_RATIO
                                            
                                            if passes_metric:
                                                # Store computed metrics and Adj. Close on the option for easy access
                                                option['calculated_rr_ratio'] = risk_reward_ratio
                                                option['annual_rr'] = annual_rr
                                                option['expected_profit'] = expected_profit
                                                option['adj_close'] = current_adj_close 
                                                filtered_options.append(option)
                                                
                                        # Only record the chain if it has at least one option that passed all filters
                                        if filtered_options:
                                            # Sort the options within the chain (highest R/R first)
                                            filtered_options.sort(key=lambda x: x['calculated_rr_ratio'], reverse=True)

                                            filtered_chains_summary.append({
                                                'expiration_date': expiration_date,
                                                'dte': dte,
                                                'filtered_options': filtered_options
                                            })

                        
                        # Store the summary data in the dictionary
                        daily_investable_data[ticker] = {
                            'adj_close': current_adj_close, 
                            'open_puts': open_puts_tracker[ticker],
                            'filtered_options_summary': filtered_chains_summary 
                        }
                        
                        # Add all filtered options from this ticker to the daily candidates list
                        for chain in filtered_chains_summary:
                            for option in chain['filtered_options']:
                                # Ensure ticker, dte, and exp date are on the option for global sorting
                                option['ticker'] = ticker
                                option['dte'] = chain['dte']
                                option['expiration_date'] = chain['expiration_date']
                                daily_trade_candidates.append(option)


                
                
                # --- Select the ABSOLUTE BEST NON-DUPLICATE, LIMIT-RESPECTING Contract of the Day ---
                
                best_contract = None
                trade_quantity = 0
                ask_at_entry_float = 0.0 # FIX: Renamed variable to reflect float status
                bid_at_entry = 0.0

                if daily_trade_candidates:
                    # Sort the ENTIRE list of candidates globally by the selected metric
                    if USE_ANNUAL_SELECTOR:
                        sort_key = lambda x: x.get('annual_rr', -float('inf'))
                    elif USE_EXPECTED_SELECTOR:
                        sort_key = lambda x: x.get('expected_profit', -float('inf'))
                    else:
                        # Default: risk/reward ratio (also used when USE_RR_SELECTOR is True)
                        sort_key = lambda x: x.get('calculated_rr_ratio', -float('inf'))

                    daily_trade_candidates.sort(key=sort_key, reverse=True)
                    
                    # Iterate through the ranked candidates to find the first non-duplicate and limit-respecting contract
                    for contract in daily_trade_candidates:
                        
                        ticker_check = contract['ticker']
                        
                        # 1. Check if the ticker has reached its per-stock limit
                        if open_puts_tracker[ticker_check] >= MAX_PUTS_PER_STOCK:
                            continue 
                        
                        # 2. Check for duplicate position
                        unique_key = (ticker_check, contract['strike'], contract['expiration_date'])
                        if unique_key not in active_position_keys:
                            
                            # --- QUANTITY CALCULATION (PREMIUM-BASED) ---
                            
                            # FIX: Ensure pBidPx and pAskPx are floats
                            try:
                                pBidPx_value = float(contract['pBidPx'])
                                pAskPx_value = float(contract['pAskPx']) 
                            except ValueError:
                                # Skip the contract if the bid price is invalid
                                continue
                                
                            premium_per_contract = pBidPx_value * 100.0
                            
                            # Calculate quantity based on max premium per trade (floored)
                            if premium_per_contract > 0:
                                qty_by_premium = math.floor(max_premium_per_trade_today / premium_per_contract)
                            else:
                                qty_by_premium = 0

                            # Determine position slot availability (positions, not contract counts)
                            remaining_account_position_slots = MAX_PUTS_PER_ACCOUNT - current_account_put_positions
                            remaining_stock_position_slots = MAX_PUTS_PER_STOCK - open_puts_tracker[ticker_check]

                            # If there are no position slots available (either account-level or per-stock), skip
                            if remaining_account_position_slots <= 0 or remaining_stock_position_slots <= 0:
                                # Cannot open a new position for this ticker due to position limits
                                continue

                            # Final quantity is solely determined by premium budget (contracts per position)
                            trade_quantity = qty_by_premium
                            
                            if trade_quantity >= 1:
                                best_contract = contract
                                # FIX: Capture Ask Price as a float for instant MTM calculation
                                ask_at_entry_float = pAskPx_value 
                                bid_at_entry = pBidPx_value
                                break # Found the best eligible contract with trade quantity >= 1
                        
                    
                    if best_contract:
                        
                        print(f"🥇 **ABSOLUTE BEST CONTRACT TODAY (Ranked by R/R Ratio):**")
                        
                        # Fetch the original delta value using the safer function
                        original_delta = best_contract.get('putDelta')
                        delta_float = safe_percentage_to_float(original_delta)
                        delta_str = f"{delta_float:.4f}" if delta_float is not None else "N/A"
                        
                        # Re-calculate values for printing
                        total_premium_collected = premium_per_contract * trade_quantity

                        # Calculate Strike/AdjClose ratio
                        adj_close = best_contract.get('adj_close')
                        strike_adj_close_ratio = (best_contract['strike'] / adj_close * 100) if adj_close and adj_close > 0 else None

                        best_info = (
                            f"  1. **{best_contract['ticker']}:** Qty={trade_quantity}, "
                            f"Total Premium Collected=${total_premium_collected:,.2f}, "
                            f"Strike=${best_contract['strike']:.2f}, "
                            f"DTE={best_contract['dte']}, "
                            f"R/R={best_contract['calculated_rr_ratio']:.2f}"
                        )
                        print(best_info)
                        
                        # Print Strike/AdjClose ratio
                        if strike_adj_close_ratio is not None:
                            print(f"     Strike/AdjClose Ratio: {strike_adj_close_ratio:.2f}%")
                        
                    else:
                        print("❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (All candidates failed limits/duplication checks or resulted in Qty=0).")
                        
                else:
                    print(f"❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (No contract passed filters).")
                
                
                # --- TRADING LOGIC: ENTER POSITION ---
                if best_contract and trade_quantity >= 1:
                    ticker_to_enter = best_contract['ticker']
                    
                    # Store current values for logging the change
                    cash_before_trade = cash_balance
                    
                    # 1. Commission Cost (Realized Loss)
                    entry_commission = trade_quantity * COMMISSION_PER_CONTRACT
                    
                    # 2. Update the master entry counters
                    total_entry_events += 1 # Increment by 1 for each new trade opened (EVENT)
                    total_contracts_opened_qty += trade_quantity # Increment by contract quantity (QTY)

                    # 3. Calculate cash change and P&L
                    daily_pnl -= entry_commission
                    cumulative_realized_pnl -= entry_commission
                    premium_inflow = premium_per_contract * trade_quantity
                    
                    # Calculate instant MTM liability and change for logging
                    position_liability_at_entry = ask_at_entry_float * trade_quantity * 100.0
                    
                    # Instantaneous MTM Change = Premium Inflow - Liability - Commission
                    instant_mtm_change = premium_inflow - position_liability_at_entry - entry_commission
                    
                    # 4. Update cash balance: Cash increases by premium inflow minus commission (Physical Cash Flow)
                    cash_balance += premium_inflow
                    cash_balance -= entry_commission 
                    
                    # 5. Update the position count
                    open_puts_tracker[ticker_to_enter] += 1
                    
                    # 6. Log the trade details (include quantity)
                    trade_entry = {
                        'entry_date': daily_date_obj.strftime('%Y-%m-%d'),
                        'ticker': ticker_to_enter,
                        'strike': best_contract['strike'],
                        'expiration_date': best_contract['expiration_date'],
                        'premium_received': bid_at_entry, 
                        'quantity': trade_quantity,
                        'unique_key': (ticker_to_enter, best_contract['strike'], best_contract['expiration_date'])
                    }
                    open_trades_log.append(trade_entry)
                    
                    # 7. Update the quick-check set
                    active_position_keys.add(trade_entry['unique_key'])
                    
                    # 8. Print the consolidated portfolio summary
                    print_daily_portfolio_summary(open_puts_tracker)
                    
                    # --- NEW: DETAILED TRANSACTION LOG ---
                    
                    print("\n📈 **TODAY'S ENTRY TRANSACTION DETAILS:**")
                    print(f"  | Ticker/Contract: {ticker_to_enter} (Qty {trade_quantity})")
                    print(f"  | Bid Price: ${bid_at_entry:.2f} | Ask Price: ${ask_at_entry_float:.2f}")
                    
                    # --- DETAILED VALUE CALCULATION INSERTED HERE (FIXED) ---
                    print("\n💵 **VALUE CALCULATION AT ENTRY (MTM):**")
                    print(f"  | Cash Balance Before: ${cash_before_trade:,.2f}")
                    print(f"  | + Gross Premium Collected (Cash Inflow): +${premium_inflow:,.2f}")
                    print(f"  | - Entry Commission: -${entry_commission:,.2f}")
                    print(f"  | - Instant MTM Liability (Cost to Close): -${position_liability_at_entry:,.2f}")
                    print(f"  | **Instantaneous Change to Portfolio Value (MTM):** ${instant_mtm_change:,.2f} (Expected small negative)")
                    print(f"  | **New Cash Balance:** ${cash_balance:,.2f} (Available for margin)")
                    # --- END DETAILED VALUE CALCULATION ---
                    
                    print(f"  | Position Liability (Ask Price): ${position_liability_at_entry:,.2f}")
            
            # ----------------------------------------------
            # --- FINAL EOD VALUATION CALCULATIONS ---
            # --- THIS BLOCK MUST RUN AFTER ALL ENTRIES/EXITS ARE COMPLETE ---
            # ----------------------------------------------
            
            # Re-initialize liability trackers to correctly sum up current state
            unrealized_pnl = 0.0 
            total_put_liability = 0.0 
            total_open_premium_collected = 0.0 
            daily_liability_itemization = [] 

            for trade in open_trades_log:
                # Use the conservative exit price (Ask price) for valuation
                current_price = get_contract_exit_price(
                    daily_orats_data, 
                    trade['ticker'], 
                    trade['expiration_date'], 
                    trade['strike']
                )
                
                if current_price is not None:
                    premium_collected_trade = trade['premium_received'] * trade['quantity'] * 100.0
                    put_cost_to_close = current_price * trade['quantity'] * 100.0
                    
                    # 1. Update total liability
                    total_put_liability += put_cost_to_close 
                    
                    # 2. Update total premium collected on open puts
                    total_open_premium_collected += premium_collected_trade
                    
                    # 3. UPnL: (Premium Collected - Cost to Close)
                    pnl_one_position = premium_collected_trade - put_cost_to_close
                    unrealized_pnl += pnl_one_position
                    
                    # 4. Itemization for Printout
                    item_detail = (
                        f"  > **{trade['ticker']}** (Qty {trade['quantity']}, Strike ${trade['strike']:.2f}, Exp {trade['expiration_date']}): "
                        f"Ask=${current_price:.2f}, Cost to Close=${put_cost_to_close:,.2f}"
                    )
                    daily_liability_itemization.append(item_detail)
                    
                else:
                    # Log the skipped contract for debugging the data source
                    if DEBUG_VERBOSE:
                        print(f"⚠️ **WARNING:** Cannot price contract {trade['ticker']} Strike ${trade['strike']:.2f}, Exp {trade['expiration_date']} for liability calculation on {date_str}. Skipping this position for today's MTM.")

            # Total Account Value (Net Asset Value)
            # CRITICAL FIX: NAV = Cash Balance - Total Cost to Close (Liability)
            # This is the industry-standard way to calculate NAV for short premium strategies.
            total_account_value = cash_balance - total_put_liability
            
            # FIX: Update the monthly log with the EOD Total Account Value.
            month_key = (daily_date_obj.year, daily_date_obj.month)
            current_pnl, _, _ = monthly_pnl_log.get(month_key, (0.0, 0.0, 0.0))
            
            # Capture MTM value and the latest SPY close price for this month
            current_spy_close = monthly_spy_prices.get(month_key, 0.0)
            monthly_pnl_log[month_key] = (current_pnl, total_account_value, current_spy_close)
            
            # Print Account Value breakdown (Corrected for Accuracy and Transparency)
            print(f"💵 **DAILY ACCOUNT VALUE (EOD - NAV):** ${total_account_value:,.2f}")            
            print(f"  > **Cash Balance:** ${cash_balance:,.2f}")
            # --- PROMOTED LIABILITY PRINT (This is the cumulative value) ---
            print(f"🛑 **TOTAL PORTFOLIO LIABILITY (Cost to Close):** ${total_put_liability:,.2f} (Computed using Ask Price)")
            
            # Print Itemized Liability Breakdown
            if daily_liability_itemization:
                for item in daily_liability_itemization:
                    print(item)

            
            print(f"  > **Total accumulated Premium on Open Puts:** +${total_open_premium_collected:,.2f}")
            # Net Unrealized P&L is still calculated using the old definition: (Premium - Liability). 
            # We display it here for informational purposes, but it is NOT used in NAV.
            print(f"  > **Net Unrealized P&L:** ${unrealized_pnl:,.2f}")

            # NEW: Current Drawdown vs. peak NAV so far
            try:
                if peak_account_value and peak_account_value > 0:
                    current_drawdown_pct = ((total_account_value / peak_account_value) - 1.0) * 100.0
                else:
                    current_drawdown_pct = 0.0
                # Track the worst drawdown seen so far
                if current_drawdown_pct < worst_drawdown_pct:
                    worst_drawdown_pct = current_drawdown_pct
                print(f"  > **Current Drawdown:** {current_drawdown_pct:.2f}% (vs peak ${peak_account_value:,.2f})")
            except Exception:
                # If any unexpected numeric issue occurs, skip printing drawdown for the day
                pass

            # Update peak NAV after computing drawdown for this day
            if total_account_value is not None:
                try:
                    if float(total_account_value) > float(peak_account_value):
                        peak_account_value = float(total_account_value)
                except Exception:
                    pass
            
            # Print Realized P&L
            if daily_pnl != 0.0:
                print(f"💸 **DAILY NET REALIZED P&L:** ${daily_pnl:,.2f}")

            print(f"💵 **TOTAL ACCOUNT VALUE (EOD - NAV):** ${total_account_value:,.2f}")    
            print("-" * 35)

            # Continue with Ticker-by-Ticker printing (optional but useful)
            if DEBUG_VERBOSE and (daily_investable_data or account_full_today):
                sorted_tickers = sorted(daily_investable_data.keys())
                
                for ticker in sorted_tickers:
                    data = daily_investable_data[ticker]
                    
                    summary_parts = []
                    total_filtered_options = 0
                    
                    # Get the single best R/R contract for this ticker for the summary line
                    ticker_best_option = get_absolute_best_contract(ticker, data['filtered_options_summary'])
                    
                    if ticker_best_option:
                        ticker_best_info = (
                            f"R/R={ticker_best_option['calculated_rr_ratio']:.2f}, "
                            f"Strike=${ticker_best_option['strike']:.2f}, "
                            f"DTE={ticker_best_option['dte']}"
                        )
                    else:
                        ticker_best_info = "N/A"
                        
                    for chain in data['filtered_options_summary']:
                        dte = chain['dte']
                        count = len(chain['filtered_options'])
                        total_filtered_options += count
                        summary_parts.append(f"{dte} days: {count} contracts")
                        
                    interval_list_str = "; deep-well".join(summary_parts) 
                    
                    # These are the lines now controlled by DEBUG_VERBOSE
                    print(
                        f"  | {ticker}: Adj.Close=${data['adj_close']:<7.2f} | Puts={data['open_puts']} | Total Viable Options: {total_filtered_options}"
                    )
                    
                    # Print the detailed breakdown if contracts were found
                    if total_filtered_options > 0:
                         print(f"  |   > Ticker's Best Contract: {ticker_best_info}")
                         print(f"  |   > Details by DTE: {interval_list_str}")
                    else:
                         print(f"  |   > Details: None Found")
                         
            # --- END DAILY PROCESSING ---
    
    # 6. Final Liquidation and Performance Summary (Runs after all trading days are processed)
    
    # --- Liquidation Preparation ---
    total_liquidation_pnl = 0.0
    
    # Check if there are positions to liquidate
    positions_to_liquidate = [trade for trade in open_trades_log if open_puts_tracker.get(trade['ticker'], 0) > 0]

    if positions_to_liquidate:
        
        print(f"\n--- Final Open Puts Tally (Tickers with Open Positions) ---")
        for ticker in sorted(open_puts_tracker.keys()):
            count = open_puts_tracker[ticker]
            if count > 0:
                print(f"  {ticker:<5}: {count} open puts (REMAINED OPEN AT LIQUIDATION)")
        
        print("\n--- FINAL PORTFOLIO LIQUIDATION ---")
        
        # Prepare header for liquidation table
        print("| Ticker | Qty | Strike   | Premium Sold  | Closing Ask  | Cost to Close  | Exit Commission | Net Gain/Loss |")
        
        # CRITICAL FIX 9: Adjust the header separator line based on the visual widths.
        print("|--------|-----|----------|---------------|--------------|----------------|-----------------|---------------|") 
        
        # We liquidate all remaining trades
        for trade in positions_to_liquidate:
            
            # Use the LAST calculated Ask Price (Liability) from the final processed day for closing
            closing_ask = get_contract_exit_price(
                last_daily_orats_data, 
                trade['ticker'], 
                trade['expiration_date'], 
                trade['strike']
            )
            
            qty = trade['quantity']
            
            if closing_ask is not None:
                
                premium_collected_per_contract = trade['premium_received']
                
                # Financials (based on last known market price)
                exit_commission = qty * FINAL_COMMISSION_PER_CONTRACT
                premium_collected_gross = premium_collected_per_contract * qty * 100.0 - qty * FINAL_COMMISSION_PER_CONTRACT
                cost_to_close_gross = closing_ask * qty * 100.0 + exit_commission
                
                
                # P&L Calculation: (Initial Premium) - (Cost to Close) - (Exit Commission)
                position_net_gain = premium_collected_gross - cost_to_close_gross 
                
                total_liquidation_pnl += position_net_gain
                
                # Accumulate gain for the Liquidation row in attribution table
                liquidation_gain += position_net_gain
                liquidation_premium_collected += premium_collected_gross
                
                # Adjust cash balance: Cash balance already holds the premium. Now we pay the cost to close.
                # Total Debit Outflow = Cost to Close + Commission
                total_debit_outflow = cost_to_close_gross # INCLUDES commission already
                
                cash_balance -= total_debit_outflow 
                # Yuda: I don;t like this cash_balance += premium_collected_gross
                
                # Calculate percentage gain relative to max risk
                max_risk_per_contract = (trade['strike'] * 100.0) - premium_collected_per_contract
                if max_risk_per_contract > 0:
                    position_gain_percent = (position_net_gain / (max_risk_per_contract * qty)) * 100.0
                else:
                    position_gain_percent = 0.0

                # --- CAPTURE COMPLETE TRADE LOG (Liquidation) ---
                # FINAL EXIT EVENT COUNT: Increment by 1 for every unique trade closed during liquidation
                total_exit_events += 1
                
                trade_to_log = {
                    'Ticker': trade['ticker'],
                    'Strike': trade['strike'],
                    'ExpDate': trade['expiration_date'],
                    'DayIn': trade['entry_date'],
                    'PriceIn': trade['premium_received'], # Option Bid Price in
                    'Qty': qty,
                    'AmountIn': premium_collected_gross, # Gross premium collected
                    'DayOut': sim_end_date.strftime('%Y-%m-%d'),
                    'PriceOut': closing_ask, # Option Ask Price at liquidation
                    'QtyOut': qty,
                    'AmountOut': cost_to_close_gross, # Gross cost to close
                    'ReasonWhyClosed': "LIQUIDATION",
                    'Gain$': position_net_gain,
                    'Gain%': position_gain_percent,
                }
                closed_trades_log.append(trade_to_log)
                
                # FIX 3 (Final Data Format): Split the $ sign and the number into separate fields to align the pipes perfectly.
                # Data widths used: Strike(6.2f), Premium(11,.2f), Ask(10.2f), Cost(12,.2f), Commission(13.2f), Net(11.2f)
                print(
                    f"| {trade['ticker']:<6} | {qty:3} | $ {trade['strike']:>6.2f} | $ {premium_collected_gross:>11,.2f} | "
                    f"$ {closing_ask:>10.2f} | $ {cost_to_close_gross:>12,.2f} | $ {exit_commission:>13.2f} | "
                    f"$ {position_net_gain:>11.2f} |"
                )
        
        # FINAL REALIZED P&L for Performance Metrics
        final_realized_profit = cumulative_realized_pnl + total_liquidation_pnl
        final_account_value_liquidated = cash_balance
        
        print("\n--- FINAL LIQUIDATION SUMMARY ---")
        print(f"💰 **FINAL REALIZED CASH VALUE:** ${final_account_value_liquidated:,.2f}")
        print(f"✅ **TOTAL LIQUIDATION P&L:** ${total_liquidation_pnl:,.2f}")
        print(f"💵 **TOTAL NET PROFIT (Start to Finish):** ${final_account_value_liquidated - INITIAL_CASH:,.2f}")
        
    else:
        # If no open positions, the final account value is the last EOD calculated value
        final_account_value_liquidated = total_account_value
        final_realized_profit = cumulative_realized_pnl
        print(f"\n--- Final Open Puts Tally (Tickers with Open Positions) ---")
        print("  (No open positions remained for liquidation.)")
        print("\n--- FINAL LIQUIDATION SUMMARY ---")
        print(f"💰 **FINAL ACCOUNT ACCOUNT VALUE (CASH):** ${final_account_value_liquidated:,.2f}")
        print(f"✅ **TOTAL LIQUIDATION P&L:** $0.00 (No open positions remained)")
        print(f"💵 **TOTAL NET PROFIT (Start to Finish):** ${final_account_value_liquidated - INITIAL_CASH:,.2f}")

    # 7. Final Performance Calculation
    
    total_sim_days = (sim_end_date - sim_start_date).days if sim_start_date and sim_end_date else 0
    total_sim_years = total_sim_days / 365.25
    
    # Total Gain Calculation
    total_net_profit = final_account_value_liquidated - INITIAL_CASH
    percent_total_gain = (total_net_profit / INITIAL_CASH) * 100.0 if INITIAL_CASH > 0 else 0.0
    
    # Annualized Gain Calculation (CAGR)
    annualized_gain = 0.0
    if total_sim_years > 0 and final_account_value_liquidated > 0 and INITIAL_CASH > 0:
        annualized_gain = (math.pow((final_account_value_liquidated / INITIAL_CASH), (1 / total_sim_years)) - 1) * 100.0
        
    # SPY Benchmark Calculation
    spy_total_return = 0.0
    spy_annualized_return = 0.0
    if spy_start_price is not None and spy_end_price is not None and spy_start_price > 0:
        spy_total_return = ((spy_end_price / spy_start_price) - 1) * 100.0
        if total_sim_years > 0:
            spy_annualized_return = (math.pow((spy_end_price / spy_start_price), (1 / total_sim_years)) - 1) * 100.0

    print(f"\n--- CUMULATIVE PERFORMANCE SUMMARY ({total_sim_days} days) ---")
    
    print(f"📈 **Simulation Period:** {sim_start_date} to {sim_end_date}")
    
    print("\n")
    print("| Metric                  |  Account Gain [%] | SPY Benchmark  | Comparison       |") 
    print("|-------------------------|-------------------|----------------|------------------|")
    # FIX 4: Aligned columns using refined explicit width and right alignment (>)
    # Portfolio Gain (16), SPY Benchmark (13), Comparison (10)
    print(f"| **Total Net Gain (%)**  | {percent_total_gain:>16.2f}% | {spy_total_return:>13.2f}% | **{percent_total_gain - spy_total_return:>10.2f}pp** |")
    print(f"| **Annualized Gain (%)** | {annualized_gain:>16.2f}% | {spy_annualized_return:>13.2f}% | **{annualized_gain - spy_annualized_return:>10.2f}pp** |")
    
    # 8. Monthly and Yearly Performance Tables
    
    # --- P&L Aggregation for Tables ---
    # Final structure: (Month) -> (% Gain, $ Gain, End Value, % SPY Gain)
    monthly_performance = {} 
    yearly_performance = {}
    
    # Pre-calculate the starting value for each month and SPY price
    sorted_months = sorted(monthly_pnl_log.keys())
    
    # --- Monthly Calculation Pass ---
    # Get SPY start price for the first month
    spy_start_of_sim = spy_start_price if spy_start_price is not None else 0.0
    
    # This will hold the closing SPY price from the last day of the *previous* month.
    spy_previous_close = spy_start_of_sim 
    
    # This will hold the MTM value from the last day of the *previous* month.
    portfolio_previous_close = INITIAL_CASH
    
    for i, (year, month) in enumerate(sorted_months):
        pnl_realized, month_end_value_mtm, spy_month_close = monthly_pnl_log[(year, month)]
        
        # 1. Portfolio Metrics
        month_start_value = portfolio_previous_close
            
        # ***CRITICAL FIX***: Replace the last month's MTM value with the final liquidiated cash value.
        is_last_month = i == len(sorted_months) - 1
        month_end_value_reported = final_account_value_liquidated if is_last_month else month_end_value_mtm
        
        # Calculate Monthly Gains
        monthly_gain_abs = month_end_value_reported - month_start_value
        monthly_gain_pct = (monthly_gain_abs / month_start_value) * 100.0 if month_start_value > 0 else 0.0
        
        # 2. SPY Metrics
        spy_monthly_return = 0.0
        if spy_month_close > 0 and spy_previous_close > 0:
            spy_monthly_return = ((spy_month_close / spy_previous_close) - 1) * 100.0
        
        monthly_performance[(year, month)] = {
            'end_value': month_end_value_reported,
            'gain_abs': monthly_gain_abs,
            'gain_pct': monthly_gain_pct,
            'spy_gain_pct': spy_monthly_return
        }
        
        # 3. Aggregate to Year (Yearly Start Value, End Value, Realized PNL)
        if year not in yearly_performance:
            yearly_performance[year] = {
                'start_value': month_start_value, # Start value of the first month in the year
                'end_value': month_end_value_reported,
                'realized_pnl': pnl_realized,
                'spy_start_price': spy_previous_close,
                'spy_end_price': spy_month_close
            }
        else:
            # Update the End Value to the current month's reported end value
            yearly_performance[year]['end_value'] = month_end_value_reported
            yearly_performance[year]['realized_pnl'] += pnl_realized
            yearly_performance[year]['spy_end_price'] = spy_month_close

        # Set values for the next iteration
        portfolio_previous_close = month_end_value_reported
        spy_previous_close = spy_month_close

    # --- Print Monthly Table ---
    print("")
    print("\n--- MONTHLY PORTFOLIO GAIN ---")
    # NEW COLUMN: % SPY Gain
    print("| Month   | Total Value EOD  | $ Gain       |  % Gain  | % SPY Gain |")
    print("|---------|------------------|--------------|----------|------------|") 
    
    for (year, month), data in monthly_performance.items():
        month_label = datetime(year, month, 1).strftime('%Y-%m')
        
        # Data widths used: End Value (11,.2f), $ Gain (9,.2f), % Gain (6.2f), % SPY Gain (8.2f)
        print(
            f"| {month_label:^5} | $ {data['end_value']:>12,.2f}   | $ {data['gain_abs']:>11,.2f} | "
            f"{data['gain_pct']:>7.2f}% | {data['spy_gain_pct']:>9.2f}% |"
        )

    # --- Print Yearly Table ---
    print("")
    print("\n--- YEARLY PORTFOLIO GAIN ---")
    # NEW COLUMN: % SPY Gain
    print("| Year    | Total Value EOD  | $ Gain        | % Gain  | % SPY Gain |")
    print("|---------|------------------|---------------|---------|------------|") 
    
    for year in sorted(yearly_performance.keys()):
        data = yearly_performance[year]
        year_end_value = data['end_value']
        year_start_value = data['start_value']
        spy_start = data['spy_start_price']
        spy_end = data['spy_end_price']
        
        yearly_gain_abs = year_end_value - year_start_value
        yearly_gain_pct = (yearly_gain_abs / year_start_value) * 100.0 if year_start_value > 0 else 0.0
        
        # Calculate Yearly SPY Gain
        spy_yearly_return = 0.0
        if spy_end > 0 and spy_start > 0:
            spy_yearly_return = ((spy_end / spy_start) - 1) * 100.0

        # Data widths used: End Value (11,.2f), $ Gain (9,.2f), % Gain (6.2f), % SPY Gain (8.2f)
        print(
            f"| {year:^5}   | $ {year_end_value:>12,.2f}   | $ {yearly_gain_abs:>11,.2f} | "
            f"{yearly_gain_pct:>6.2f}% | {spy_yearly_return:>9.2f}% |"
        )

    # 9. Exit Statistics (Focusing on Entry/Exit Events)
    total_closed_positions_qty = stop_loss_count + expired_otm_count + expired_itm_count
    
    # Calculate Total Exit Events (Total number of discrete trades closed)
    total_exit_events_count = len(closed_trades_log)

    print("")
    print("\n--- TRADE EXIT STATISTICS (by Trade Event Count) ---")
    
    # Define Total Gain/Premium Collected for the whole simulation
    TOTAL_GAIN = stop_loss_gain + expired_otm_gain + expired_itm_gain + liquidation_gain
    TOTAL_PREMIUM_COLLECTED = stop_loss_premium_collected + expired_otm_premium_collected + expired_itm_premium_collected + liquidation_premium_collected
    
    # --- CRITICAL FIX: Define Event Counters here ---
    # Calculate Exit Event Counts: Use substring matching to be robust to slightly different reason text.
    # Treat any reason containing 'STOP LOSS' as a stop-loss event (covers position-level and stock-level stop losses).
    stop_loss_events = sum(
        1 for trade in closed_trades_log
        if trade.get('ReasonWhyClosed') and 'STOP LOSS' in trade.get('ReasonWhyClosed')
    )

    expired_otm_events = sum(
        1 for trade in closed_trades_log
        if trade.get('ReasonWhyClosed') and 'Expiration (OTM' in trade.get('ReasonWhyClosed')
    )

    expired_itm_events = sum(
        1 for trade in closed_trades_log
        if trade.get('ReasonWhyClosed') and 'Expiration (ITM' in trade.get('ReasonWhyClosed')
    )

    liquidation_events = sum(
        1 for trade in closed_trades_log
        if trade.get('ReasonWhyClosed') and 'LIQUIDATION' in trade.get('ReasonWhyClosed')
    )

    # Total Closed Trades = length of the closed trades log (more robust than summing categories)
    total_closed_events = len(closed_trades_log)

    # Calculate Net % Gain relative to premium collected for each category
    def calculate_net_gain_percent(gain, premium):
        # We calculate P&L / Premium Collected (Return on Premium)
        return (gain / premium) * 100.0 if premium != 0.0 else 0.0
    
    # Header now includes the new gain columns
    print("| Exit Reason                   | Exit Events       |  % of Total Events | Total Gain $      | Net Gain % |")
    print("|-------------------------------|-------------------|--------------------|-------------------|------------|")
    
    # 1. STOP LOSS
    stop_loss_gain_pct = calculate_net_gain_percent(stop_loss_gain, stop_loss_premium_collected)
    print(
        f"| **Stop Loss**{' ':17}| {stop_loss_events:>16,}  | {stop_loss_events / total_closed_events * 100 if total_closed_events > 0 else 0:>17.2f}% | "
        f"${stop_loss_gain:>16,.2f} | {stop_loss_gain_pct:>9.2f}% |"
    )

    # 2. EXPIRED OTM (Max Profit)
    expired_otm_gain_pct = calculate_net_gain_percent(expired_otm_gain, expired_otm_premium_collected)
    print(
        f"| **Expired OTM (Max Profit)**{' ':2}| {expired_otm_events:>16,}  | {expired_otm_events / total_closed_events * 100 if total_closed_events > 0 else 0:>17.2f}% | "
        f"${expired_otm_gain:>16,.2f} | {expired_otm_gain_pct:>9.2f}% |"
    )
    
    # 3. EXPIRED ITM (Assignment)
    expired_itm_gain_pct = calculate_net_gain_percent(expired_itm_gain, expired_itm_premium_collected)
    print(
        f"| **Expired ITM (Assignment)**  | {expired_itm_events:>16,}  | {expired_itm_events / total_closed_events * 100 if total_closed_events > 0 else 0:>17.2f}% | "
        f"${expired_itm_gain:>16,.2f} | {expired_itm_gain_pct:>9.2f}% |"
    )

    # 4. LIQUIDATION
    liquidation_gain_pct = calculate_net_gain_percent(liquidation_gain, liquidation_premium_collected)
    print(
        f"| **Liquidation**{' ':15}| {liquidation_events:>16,}  | {liquidation_events / total_closed_events * 100 if total_closed_events > 0 else 0:>17.2f}% | "
        f"${liquidation_gain:>16,.2f} | {liquidation_gain_pct:>9.2f}% |"
    )
    
    # 5. Total Exit Events Summary (Total of rows 1-4)
    print("|-------------------------------|-------------------|--------------------|-------------------|------------|")    
    # Total Premium Collected is ONLY used for Net Gain %, so we use 'N/A' for the gain %.
    print(f"| **Total Exit Trades Closed**  | {total_closed_events:>16,}  | {100.0:>17.2f}% | "
          f"${TOTAL_GAIN:>16,.2f} | {'N/A':>10} |"
    )
    
    # Row 6: Total Entry Events (For direct comparison)
    print("|-------------------------------|-------------------|--------------------|-------------------|------------|")    
    # Text length: "Total Entry Events" is 18 characters. Padding needed: 30 - 18 = 12 spaces.
    print(f"| Total Entry Events{' ':12}| {total_entry_events:>16,}  | {'N/A':>18} | {'N/A':>17} | {'N/A':>10} |")
    
    
    # 10. NEW: Detailed Closed Trade Log
    if closed_trades_log:
        
        # Sort the log by exit date
        closed_trades_log.sort(key=lambda x: x['DayOut'])

        print(f"✅ Position Stop Loss Threshold = {POSITION_STOP_LOSS_PCT * 100:.2f}%")
       

        # Adjusted separator for new Exit # column
        print("\n\n--- DETAILED CLOSED TRADE LOG (Full History) ---")
        print("| Exit #  | Ticker |  Qty |   Day In   | Price In   | Amount In  |  Day Out   | Price Out |  Amount Out | Reason Why Closed          |    Gain $  |   Gain % |")
        print("|---------|--------|------|------------|------------|------------|------------|-----------|-------------|----------------------------|------------|----------|")
        
        for index, trade in enumerate(closed_trades_log):
            
            # Exit Number (1, 2, 3...)
            exit_number = index + 1
            
            # Format numbers (Price In/Out, Amount In/Out, Gain $)
            price_in_str = f"${trade['PriceIn']:>9.2f}"
            price_out_str = f"${trade['PriceOut']:>8.2f}" if trade['PriceOut'] is not None else ""
            
            amount_in_str = f"${trade['AmountIn']:>9,.2f}"
            amount_out_str = f"${trade['AmountOut']:>10,.2f}"
            
            gain_abs_str = f"{trade['Gain$']:>10.2f}"
            gain_pct_str = f"{trade['Gain%']:>7.2f}%"
            
            # Truncate reason if necessary (Reason is 26 chars)
            # Define Column Widths
            COL_EXIT_NUM = 7
            COL_TICKER, COL_QTY, COL_ENTRY_PRICE, COL_EXIT_PRICE = 6, 4, 9, 9
            COL_IN_AMT, COL_OUT_AMT, COL_GAIN_ABS, COL_GAIN_PCT = 11, 11, 8, 7
            COL_DAY, COL_REASON = 10, 26
            reason_str = trade['ReasonWhyClosed'][:25]
            
            row = (
                f"| {exit_number:>{COL_EXIT_NUM}} | {trade['Ticker']:<{COL_TICKER}} | {trade['Qty']:>{COL_QTY}} | {trade['DayIn']:^{10}} | "
                f"{price_in_str} | {amount_in_str} | {trade['DayOut']:^{10}} | "
                f"{price_out_str} | {amount_out_str} | "
                f" {reason_str:<{25}} | {gain_abs_str} | {gain_pct_str} |"
            )
            print(row)

    print("\n=== FINAL TRADING RULES SUMMARY ===\n")
    
    # 1. Account Simulation Rules
    print("📊 Account Simulation Rules")
    print("|--------------------|----------------|")
    print("| Parameter          | Value          |")
    print("|--------------------|----------------|")
    print(f"| Start Date         | {start_date_str:<14} |")
    print(f"| Initial Cash       | ${float(rules['account_simulation']['initial_cash']):>13,.2f} |")
    print(f"| Max Puts/Account   | {MAX_PUTS_PER_ACCOUNT:>14} |")
    print(f"| Max Puts/Stock     | {MAX_PUTS_PER_STOCK:>14} |")
    print("|--------------------|----------------|")
    print()   

    # 2. Underlying Stock Rules (Final Summary)
    print("🧩 Underlying Stock Rules")
    print("|----------------------------|----------------|")
    print("| Parameter                  | Value          |")
    print("|----------------------------|----------------|")
    print(f"| Min 5-Day Rise             | {min_rise_str} |")
    print(f"| Min Above Avg              | {min_above_str} |")
    print(f"| Max Above Avg              | {max_above_str} |")
    print(f"| Min 10-Day Avg Slope       | {min_slope_str} |")
    print(f"| Min Stock Price            | {min_price_str} |")
    print("|----------------------------|----------------|")
    print()

    # 3. Entry Put Position Rules
    print("📈 Entry Put Position Rules")
    print("|------------------------|----------------|")
    print("| Parameter              | Value          |")
    print("|------------------------|----------------|")
    print(f"| Min DTE                | {MIN_DTE:>14} |")
    print(f"| Max DTE                | {MAX_DTE:>14} |")
    print(f"| Min Put Bid Price      | $ {MIN_BID_PRICE:>12.2f} |")
    print(f"| Min Put Delta          | {MIN_DELTA*100:>13.1f}% |")
    print(f"| Max Put Delta          | {MAX_DELTA*100:>13.1f}% |")
    print(f"| Max Bid-Ask Spread     | {MAX_SPREAD_DECIMAL*100:>13.1f}% |")
    print(f"| Min Avg Above Strike   | {MIN_AVG_ABOVE_STRIKE_PCT*100:>13.1f}% |")
    print(f"| Min Risk/Reward Ratio  | {MIN_RISK_REWARD_RATIO:>14.1f} |")
    print(f"| Min Annual Risk        | {safe_percentage_to_float(rules['entry_put_position']['min_annual_risk'])*100:>13.1f}% |")
    print(f"| Min Expected Profit    | {safe_percentage_to_float(rules['entry_put_position']['min_expected_profit'])*100:>13.1f}% |")
    print(f"| Use Risk/Reward Ratio  | {('Yes' if rules['entry_put_position']['select_by_risk_reward_ratio'] else 'No'):>14} |")
    print(f"| Use Annual Risk        | {('Yes' if rules['entry_put_position']['select_by_annual_risk'] else 'No'):>14} |")
    print(f"| Use Expected Profit    | {('Yes' if rules['entry_put_position']['select_by_expected_profit'] else 'No'):>14} |")
    print("|------------------------|----------------|")
    print()

    # 4. Exit Put Position Rules
    print("📉 Exit Put Position Rules")
    print("|--------------------|--------------|")
    print("| Parameter          | Value        |")
    print("|--------------------|--------------|")
    print(f"| Position Stop Loss | {POSITION_STOP_LOSS_PCT*100:>11.1f}% |")
    print(f"| Stock Below SMA150 | {STOCK_MAX_BELOW_AVG_PCT*100:>11.1f}% |")
    print("|--------------------|--------------|")
    print()

    # 4. Trading Costs and Limits
    print("💰 Trading Parameters")
    print("|--------------------|--------------|")
    print("| Parameter          | Value        |")
    print("|--------------------|--------------|")
    print(f"| Commission/Contract| ${COMMISSION_PER_CONTRACT:>11.2f} |")
    print(f"| Max Premium/Trade  | ${MAX_PREMIUM_PER_TRADE:>11.2f} |")
    print("|--------------------|--------------|")
    print() 
 
    # Compute runtime for the Performance Summary
    try:
        _elapsed_seconds = int(time.perf_counter() - _sim_start_time)
        _hh = _elapsed_seconds // 3600
        _mm = (_elapsed_seconds % 3600) // 60
        _ss = _elapsed_seconds % 60
        runtime_str = f"{_hh:02d}:{_mm:02d}:{_ss:02d}"
    except Exception:
        runtime_str = "N/A"

    # Performance Summary
    print("📊 Final Performance")
    print("|--------------------|---------------|")
    print("| Parameter          |  Value        |")
    print("|--------------------|---------------|")
    print(f"| Annualized Gain    | {annualized_gain:>12.2f}% |")
    print(f"| Total Gain         | ${TOTAL_GAIN:>12,.2f} |")
    print(f"| Run Time           | {runtime_str:>12} |")
    # NEW: Worst drawdown across all simulated dates
    try:
        print(f"| Worst Drawdown     | {worst_drawdown_pct:>12.2f}% |")
    except Exception:
        # If for any reason the metric isn't available, skip gracefully
        pass
    print("|--------------------|---------------|")
    print()    
    
# Execute the main function
if __name__ == "__main__":
    load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)