import orjson
import os
from datetime import datetime
import math
import sys
import time

# --- Logger Class ---
class Logger:
    """Redirects print statements to both the console and a log file."""
    def __init__(self, filepath, minimal_mode=False):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')
        self.minimal_mode = minimal_mode
        self.suppress_output = minimal_mode  # Start suppressed if minimal mode

    def write(self, message):
        if not self.suppress_output:
            self.terminal.write(message)
        # Always write to logfile
        self.logfile.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command, which is called by print().
        if not self.suppress_output:
            self.terminal.flush()
        self.logfile.flush()
    
    def enable_output(self):
        """Enable output to console and log file."""
        self.suppress_output = False
    
    def force_print(self, message):
        """Force print to both terminal and log, bypassing suppression."""
        self.terminal.write(message + '\n')
        self.terminal.flush()
        self.logfile.write(message + '\n')
        self.logfile.flush()

    def close(self):
        self.logfile.close()# --- Configuration ---
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
    # Preload rules to check minimal_mode setting
    try:
        with open(rules_file_path, 'rb') as f:
            rules_preview = orjson.loads(f.read())
            minimal_mode = bool(rules_preview.get("account_simulation", {}).get("Minimal_Print_Out", False))
    except:
        minimal_mode = False
    
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Timestamped log filename: yyyy-mm-dd hh-mm.log (Windows-safe: ':' not allowed)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    base_name = f"{timestamp}.log"
    log_file_path = os.path.join(LOG_DIR, base_name)
    # Ensure uniqueness if multiple runs occur within the same minute
    suffix = 1
    while os.path.exists(log_file_path):
        suffix += 1
        log_file_path = os.path.join(LOG_DIR, f"{timestamp}_{suffix}.log")
    
    original_stdout = sys.stdout
    logger = Logger(log_file_path, minimal_mode=minimal_mode)
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
        with open(rules_file_path, 'rb') as f:
            rules = orjson.loads(f.read())
            
            # --- ACCOUNT LIMITS ---
            INITIAL_CASH = float(rules["account_simulation"]["initial_cash"].replace('$', '').replace(',', '').strip())
            MAX_PUTS_PER_ACCOUNT = int(rules["account_simulation"]["max_puts_per_account"])
            MAX_PUTS_PER_STOCK = int(rules["account_simulation"]["max_puts_per_stock"])
            MAX_PUTS_PER_DAY = int(rules["account_simulation"]["max_puts_per_day"])
            MINIMAL_PRINT_OUT = bool(rules["account_simulation"].get("Minimal_Print_Out", False))
            
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
            # Ranking flags (renamed from select_by to rank_by for clarity)
            RANK_BY_RR = bool(rules['entry_put_position'].get('rank_by_risk_reward_ratio', False))
            RANK_BY_ANNUAL = bool(rules['entry_put_position'].get('rank_by_annual_risk', False))            
            RANK_BY_EXPECTED = bool(rules['entry_put_position'].get('rank_by_expected_profit', False))

            MIN_ANNUAL_RISK = float(rules['entry_put_position']['min_annual_risk_reward_ratio'])
            
            MIN_EXPECTED_PROFIT = safe_percentage_to_float(rules['entry_put_position']['min_expected_profit'])
            
            # Validate that only one ranking method is enabled at a time
            rankers_enabled = sum([1 if RANK_BY_RR else 0,
                                   1 if RANK_BY_ANNUAL else 0,
                                   1 if RANK_BY_EXPECTED else 0])
            if rankers_enabled > 1:
                print("❌ Rule error: More than one ranking method is enabled in rules.json. Please enable only one of rank_by_risk_reward_ratio, rank_by_annual_risk, rank_by_expected_profit.")
                return
            if rankers_enabled == 0:
                print("❌ Rule error: No ranking method is enabled in rules.json. Please enable one of rank_by_risk_reward_ratio, rank_by_annual_risk, rank_by_expected_profit.")
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

            # New Exit Rule: Min Gain to Take Profit (percentage profit threshold vs entry bid)
            try:
                _tp_raw = rules.get('exit_put_position', {}).get('min_gain_to_take_profit', None)
                TAKE_PROFIT_MIN_GAIN_PCT = safe_percentage_to_float(_tp_raw) if _tp_raw is not None else None
            except Exception:
                TAKE_PROFIT_MIN_GAIN_PCT = None

            # New Exit Rule: Stock Min Above Strike (percentage buffer above strike required)
            try:
                _min_above_strike_raw = rules.get('exit_put_position', {}).get('stock_min_above_strike', None)
                STOCK_MIN_ABOVE_STRIKE_PCT = safe_percentage_to_float(_min_above_strike_raw) if _min_above_strike_raw is not None else None
            except Exception:
                STOCK_MIN_ABOVE_STRIKE_PCT = None

            # New Exit Rule: Stock Max Below Entry (percentage drop from entry AdjClose)
            try:
                _max_below_entry = None
                exit_rules = rules.get('exit_put_position', {})
                if 'stock_max_below_entry' in exit_rules:
                    _max_below_entry = exit_rules.get('stock_max_below_entry')
                elif 'Stock max below entry' in exit_rules:
                    _max_below_entry = exit_rules.get('Stock max below entry')
                STOCK_MAX_BELOW_ENTRY_PCT = abs(safe_percentage_to_float(_max_below_entry)) if _max_below_entry is not None else None
            except Exception:
                STOCK_MAX_BELOW_ENTRY_PCT = None

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
            print(f"📊 Account Simulation Rules")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Start Date                 | {start_date_str:<14} |")
            print(f"| Initial Cash               | ${float(rules['account_simulation']['initial_cash']):>13,.2f} |")
            print(f"| Max Puts/Account           | {MAX_PUTS_PER_ACCOUNT:>14} |")
            print(f"| Max Puts/Stock             | {MAX_PUTS_PER_STOCK:>14} |")
            print(f"| Max Puts/Day               | {MAX_PUTS_PER_DAY:>14} |")
            print(f"|----------------------------|----------------|")
            print()
            
            # 2.b Underlying Stock Rules (precomputed values)
            print("🧩 Underlying Stock Rules")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Min 5-Day Rise             | {min_rise_str} |")
            print(f"| Min Above Avg              | {min_above_str} |")
            print(f"| Max Above Avg              | {max_above_str} |")
            print(f"| Min 10-Day Avg Slope       | {min_slope_str} |")
            print(f"| Min Stock Price            | {min_price_str} |")
            print(f"|----------------------------|----------------|")
            print()

            # 3. Entry Put Position Rules
            print("📈 Entry Put Position Rules")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Min DTE                    | {MIN_DTE:>14} |")
            print(f"| Max DTE                    | {MAX_DTE:>14} |")
            print(f"| Min Put Bid Price          | $ {MIN_BID_PRICE:>12.2f} |")
            print(f"| Min Put Delta              | {MIN_DELTA*100:>13.1f}% |")
            print(f"| Max Put Delta              | {MAX_DELTA*100:>13.1f}% |")
            print(f"| Max Bid-Ask Spread         | {MAX_SPREAD_DECIMAL*100:>13.1f}% |")
            print(f"| Min Avg Above Strike       | {MIN_AVG_ABOVE_STRIKE_PCT*100:>13.1f}% |")
            print(f"| Min Risk/Reward Ratio      | {MIN_RISK_REWARD_RATIO:>14.1f} |")
            print(f"| Min Annual Risk            | {MIN_ANNUAL_RISK:>14.1f} |")
            print(f"| Min Expected Profit        | {MIN_EXPECTED_PROFIT*100:>13.1f}% |")
            print(f"| Rank By Risk/Reward Ratio  | {('Yes' if RANK_BY_RR else 'No'):>14} |")
            print(f"| Rank By Annual Risk        | {('Yes' if RANK_BY_ANNUAL else 'No'):>14} |")
            print(f"| Rank By Expected Profit    | {('Yes' if RANK_BY_EXPECTED else 'No'):>14} |")
            print(f"|----------------------------|----------------|")
            print()

            # 4. Exit Put Position Rules
            print(f"📉 Exit Put Position Rules")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Position Stop Loss         | {POSITION_STOP_LOSS_PCT*100:>13.1f}% |")
            print(f"| Stock Below SMA150         | {STOCK_MAX_BELOW_AVG_PCT*100:>13.1f}% |")
            print(f"| Stock Min Above Strike     | {STOCK_MIN_ABOVE_STRIKE_PCT*100:>13.1f}% |")   
            print(f"| Stock Max Below Entry      | {STOCK_MAX_BELOW_ENTRY_PCT*100:>13.1f}% |") 
            print(f"| Min Gain to Take Profit    | {TAKE_PROFIT_MIN_GAIN_PCT*100:>13.1f}% |")
            print(f"|----------------------------|----------------|")
            print()                        

            # 4a. Trading Costs and Limits
            print("💰 Trading Parameters")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Commission/Contract        | ${COMMISSION_PER_CONTRACT:>13.2f} |")
            print(f"| Max Premium/Trade          | ${MAX_PREMIUM_PER_TRADE:>13.2f} |")
            print(f"|----------------------------|----------------|")
            print()
            
            # Stop here to view the tables
            # quit()
            # Position stop-loss rule: threshold is compared against daily option BID vs the entry BID
            # The rule is defined in `exit_put_position.position_stop_loss_pct` in rules.json
            _pos_raw = rules.get('exit_put_position', {}).get('position_stop_loss_pct', "0%")
            POSITION_STOP_LOSS_PCT = abs(safe_percentage_to_float(_pos_raw)) if _pos_raw is not None else 0.0

    except Exception as e:
        print(f"❌ Error loading/parsing rules.json values: {e}")
        return
    
    # 2. Load the main ticker data from stock_history.json
    print("Loading stock_history.json")
    try:
        with open(json_file_path, 'rb') as f:
            stock_history_dict = orjson.loads(f.read())
    except FileNotFoundError:
        print(f"❌ Error: The data file '{json_file_path}' was was not found.")
        return
    except orjson.JSONDecodeError:
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

    # NEW: Win ratio counters (avoid end-of-run scan)
    winning_trades_count = 0
    closed_trades_count = 0

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
    # NEW: Take-Profit aggregators
    take_profit_gain = 0.0
    take_profit_premium_collected = 0.0

    # NEW: Track peak NAV to compute current drawdown
    peak_account_value = INITIAL_CASH
    # NEW: Track worst drawdown percentage observed across all simulated dates (negative number, e.g., -12.34)
    worst_drawdown_pct = 0.0
    # NEW: Track peak number of open positions across the account
    peak_open_positions = 0

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
    
    # Track last printed month for monthly progress reports
    last_printed_month = None
    last_monthly_summary_msg = None
    
    # Convert to list and filter for dates >= start_date
    all_dates_list = list(sorted_unique_dates)
    
    for idx, date_str in enumerate(all_dates_list):
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
                with open(orats_file_path, 'rb') as f:
                    daily_orats_data = orjson.loads(f.read())
                    last_daily_orats_data = daily_orats_data # Store the latest successful load
            except (FileNotFoundError, orjson.JSONDecodeError):
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
            
            # Track number of new positions entered today
            daily_entries_count = 0

            # ----------------------------------------------------
            # --- Position Management/Exit Logic (Stop-Loss & Expiration) ---
            # ----------------------------------------------------
            positions_to_remove = []

            # Precompute entry-drop breached tickers (if rule configured)
            entry_drop_breached_tickers = set()
            if 'STOCK_MAX_BELOW_ENTRY_PCT' in locals() and STOCK_MAX_BELOW_ENTRY_PCT is not None and STOCK_MAX_BELOW_ENTRY_PCT > 0:
                # Build today's adj_close per ticker
                todays_adj = {}
                for tkr in stock_history_dict.keys():
                    try:
                        todays_adj[tkr] = stock_history_dict.get(tkr, {}).get(date_str, {}).get('adj_close')
                    except Exception:
                        todays_adj[tkr] = None
                # Evaluate per open trade; if any position's entry baseline breached, mark ticker to close all
                for tr in open_trades_log:
                    tkr = tr.get('ticker')
                    entry_base = tr.get('entry_adj_close')
                    curr_px = todays_adj.get(tkr)
                    if entry_base is None or curr_px is None:
                        continue
                    threshold_px = entry_base * (1.0 - STOCK_MAX_BELOW_ENTRY_PCT)
                    if curr_px < threshold_px:
                        entry_drop_breached_tickers.add(tkr)

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
                strike_buffer_stop_triggered = False
                entry_drop_stop_triggered = False
                take_profit_triggered = False
                if current_adj_close is not None and sma150_adj_close is not None:
                    
                    # Calculate the threshold: SMA150 * (1 - Max Drop %)
                    threshold = sma150_adj_close * (1.0 - STOCK_MAX_BELOW_AVG_PCT)
                    
                    if current_adj_close < threshold:
                        stop_loss_triggered = True

                # 1.a New stock-vs-strike buffer stop rule (per-position)
                # If configured, exit when current AdjClose is below Strike * (1 + buffer_pct)
                try:
                    if current_adj_close is not None and STOCK_MIN_ABOVE_STRIKE_PCT is not None:
                        strike_buffer_threshold = trade['strike'] * (1.0 + STOCK_MIN_ABOVE_STRIKE_PCT)
                        if current_adj_close < strike_buffer_threshold:
                            strike_buffer_stop_triggered = True
                            stop_loss_triggered = True
                            # Ensure we have an exit price; prefer ASK, then BID as fallback
                            if current_ask_price is None:
                                fallback_ask2 = get_contract_exit_price(
                                    daily_orats_data,
                                    trade['ticker'],
                                    trade['expiration_date'],
                                    trade['strike']
                                )
                                if fallback_ask2 is not None:
                                    current_ask_price = fallback_ask2
                except Exception:
                    pass

                # 1.b New stock-vs-entry drop rule (per-ticker close-all)
                if 'entry_drop_breached_tickers' in locals() and ticker in entry_drop_breached_tickers:
                    entry_drop_stop_triggered = True
                    stop_loss_triggered = True
                    # Ensure we have an exit price; prefer ASK, fallback to BID
                    if current_ask_price is None:
                        try:
                            fallback_ask3 = get_contract_exit_price(
                                daily_orats_data,
                                trade['ticker'],
                                trade['expiration_date'],
                                trade['strike']
                            )
                            if fallback_ask3 is not None:
                                current_ask_price = fallback_ask3
                        except Exception:
                            pass

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

                # 1.c TAKE-PROFIT CHECK (based on current ASK vs entry BID)
                try:
                    if (
                        TAKE_PROFIT_MIN_GAIN_PCT is not None and TAKE_PROFIT_MIN_GAIN_PCT > 0 and
                        entry_bid_price is not None and entry_bid_price > 0
                    ):
                        # Ensure we have an ask price to compute profit; prefer ASK
                        if current_ask_price is None:
                            fallback_tp_ask = get_contract_exit_price(
                                daily_orats_data,
                                trade['ticker'],
                                trade['expiration_date'],
                                trade['strike']
                            )
                            current_ask_price = fallback_tp_ask
                        if current_ask_price is not None and current_ask_price > 0:
                            profit_ratio = 1.0 - (current_ask_price / entry_bid_price)
                            if profit_ratio >= TAKE_PROFIT_MIN_GAIN_PCT:
                                take_profit_triggered = True
                except Exception:
                    pass
                        
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


                if stop_loss_triggered or take_profit_triggered or expired_triggered:
                    
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
                    # Net premium collected, subtracting entry commission
                    premium_collected_gross = (trade['premium_received'] * qty * 100.0)
                    entry_commission = qty * COMMISSION_PER_CONTRACT
                    
                    if expired_triggered:
                        # --- CRITICAL FIX FOR OTM/ITM ASSIGNMENT ---
                        
                        if current_adj_close is None:
                            # CRITICAL BUG FIX: Don't silently skip expired positions!
                            # If we can't get stock price, we must still close the position
                            # and record it with a warning
                            print(f"⚠️ **EXPIRED POSITION - NO STOCK PRICE:** {daily_date_obj}")
                            print(f"    Symbol: {trade['ticker']}")
                            print(f"    Strike: ${trade['strike']:.2f}")
                            print(f"    Expiration: {trade['expiration_date']}")
                            print(f"    WARNING: Missing adj_close data - assuming OTM expiration")
                            
                            # Treat as OTM expiration (conservative assumption)
                            cost_to_close_gross = 0.0 
                            exit_commission = 0.0
                            net_profit = premium_collected_gross - entry_commission - exit_commission
                            
                            expired_otm_count += 1
                            expired_otm_gain += net_profit
                            expired_otm_premium_collected += premium_collected_gross
                            
                            exit_details['PriceOut'] = 0.0
                            exit_details['AmountOut'] = 0.0
                            exit_details['ReasonWhyClosed'] = "Expiration (OTM/No Price Data)"
                            
                            # Continue with the exit flow instead of skipping
                            # (The rest of the exit code will handle logging and removal)
                        else:
                            # We have stock price data - proceed with normal ITM/OTM logic
                            is_itm = current_adj_close < trade['strike']
                            
                            if is_itm:
                                # ITM/ASSIGNMENT SCENARIO (Loss)
                                assignment_loss_gross = (trade['strike'] - current_adj_close) * qty * 100.0 + entry_commission + exit_commission 
                                net_profit = premium_collected_gross  - assignment_loss_gross - exit_commission
                                
                                # Count exit EVENTS (not contracts)
                                expired_itm_count += 1
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
                                net_profit = premium_collected_gross- entry_commission - exit_commission
                                
                                # Count exit EVENTS (not contracts)
                                expired_otm_count += 1
                                expired_otm_gain += net_profit
                                expired_otm_premium_collected += premium_collected_gross
                                
                                exit_details['PriceOut'] = 0.0 # Option price at expiration
                                exit_details['AmountOut'] = 0.0 # Gross cost to close
                                exit_details['ReasonWhyClosed'] = "Expiration (OTM/Max Profit)"
                            
                        # --- END CRITICAL FIX --- 
                        
                    elif (stop_loss_triggered or take_profit_triggered) and current_ask_price is not None:
                        # STOP LOSS or TAKE PROFIT SCENARIO (uses option Ask for closure)
                        cost_to_close_gross = current_ask_price * qty * 100.0 + exit_commission
                        net_profit = premium_collected_gross - cost_to_close_gross - exit_commission

                        # Determine specific reason (take-profit has priority if triggered)
                        if take_profit_triggered:
                            reason = "TakeProfit Min gain reached"
                            # Count take-profit aggregates
                            take_profit_gain += net_profit
                            take_profit_premium_collected += premium_collected_gross
                        else:
                            if 'position_stop_loss_triggered' in locals() and position_stop_loss_triggered:
                                reason = "StopLoss Bid above entry threshold"
                            elif strike_buffer_stop_triggered:
                                reason = "StopLoss Stk below Strike Threshold"
                            elif entry_drop_stop_triggered:
                                reason = "StopLoss Stk Below Entry Threshold)"
                            else:
                                reason = "StopLoss Stk Below SMA150"

                            # Only count toward stop-loss stats when it's a stop-loss exit
                            # Count exit EVENTS (not contracts)
                            stop_loss_count += 1
                            stop_loss_gain += net_profit
                            stop_loss_premium_collected += premium_collected_gross

                        exit_details['PriceOut'] = current_ask_price # Option Ask Price at Exit
                        exit_details['AmountOut'] = cost_to_close_gross # Gross cost to close
                        exit_details['ReasonWhyClosed'] = reason
                        
                    elif (stop_loss_triggered or take_profit_triggered) and current_ask_price is None:
                        # FALLBACK: Exit triggered but no current price - use stored last_known_ask
                        fallback_price = trade.get('last_known_ask')
                        fallback_date = trade.get('last_ask_date', 'unknown')
                        
                        if fallback_price is not None:
                            # Use the stored historical price to close the position
                            cost_to_close_gross = fallback_price * qty * 100.0 + exit_commission
                            net_profit = premium_collected_gross - cost_to_close_gross - exit_commission
                            
                            # Determine base reason
                            if take_profit_triggered:
                                base_reason = "TakeProfit"
                                take_profit_gain += net_profit
                                take_profit_premium_collected += premium_collected_gross
                            else:
                                if 'position_stop_loss_triggered' in locals() and position_stop_loss_triggered:
                                    base_reason = "StopLoss Bid above entry threshold"
                                elif strike_buffer_stop_triggered:
                                    base_reason = "StopLoss Stk below Strike Threshold"
                                elif entry_drop_stop_triggered:
                                    base_reason = "StopLoss Stk Below Entry Threshold)"
                                else:
                                    base_reason = "StopLoss Stk Below SMA150"
                                
                                stop_loss_count += 1
                                stop_loss_gain += net_profit
                                stop_loss_premium_collected += premium_collected_gross
                            
                            reason = f"{base_reason} (Price from {fallback_date})"
                            exit_details['PriceOut'] = fallback_price
                            exit_details['AmountOut'] = cost_to_close_gross
                            exit_details['ReasonWhyClosed'] = reason
                            
                            # Log the fallback usage
                            print(f"📊 **EXIT WITH STORED PRICE:** {base_reason}")
                            print(f"    {trade['ticker']} Strike ${trade['strike']:.2f}, Exp {trade['expiration_date']}, Qty {qty}")
                            print(f"    Using stored ask price ${fallback_price:.2f} from {fallback_date} (current date {date_str} has no price data)")
                        else:
                            # No stored price available - this should be very rare
                            print(f"⚠️ **EXIT FAILED - NO PRICE DATA:** {daily_date_obj}")
                            print(f"    Symbol: {trade['ticker']}")
                            print(f"    Strike: ${trade['strike']:.2f}")
                            print(f"    Expiration: {trade['expiration_date']}")
                            print(f"    Reason: {reason if 'reason' in locals() else 'Stop Loss or Take Profit'}")
                            print(f"    Current Ask price: None")
                            print(f"    Stored last_known_ask: None")
                            continue
                            
                    else:
                        # No exit condition met - skip
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
                    daily_pnl += net_profit  # Add to daily P&L only; cumulative is updated at end of day
                    
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
                        'unique_key': trade['unique_key'], # Add unique_key for validation
                        **exit_details # Merge in all exit details
                    }
                    closed_trades_log.append(trade_to_log)
                    # Update win/loss counters
                    closed_trades_count += 1
                    if net_profit > 0:
                        winning_trades_count += 1
                    
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
            
            # INTEGRITY CHECK: Store current open positions for reconciliation
            prev_open_positions = {
                trade['unique_key']: trade for trade in open_trades_log
            }
            
            # Remove closed positions from the log, iterating backwards
            for index in sorted(positions_to_remove, reverse=True):
                open_trades_log.pop(index)
            
            # Recalculate current_account_put_positions after exits
            current_account_put_positions = sum(open_puts_tracker.values())
            
            # INTEGRITY CHECK: Verify all position transitions are accounted for
            current_open_positions = {
                trade['unique_key']: trade for trade in open_trades_log
            }
            
            # Find positions that disappeared without being properly closed
            for key, trade in prev_open_positions.items():
                if (key not in current_open_positions and 
                    key not in {t['unique_key'] for t in closed_trades_log}):
                    # Position disappeared without proper closure - force close it
                    print(f"⚠️ **WARNING: Found unclosed position:** {trade['ticker']} (Strike ${trade['strike']:.2f}, Qty {trade['quantity']})")
                    
                    # Create exit record with warning
                    exit_details = {
                        'DayOut': daily_date_obj.strftime('%Y-%m-%d'),
                        'PriceOut': 0.0,  # No price available
                        'QtyOut': trade['quantity'],
                        'AmountOut': 0.0,
                        'ReasonWhyClosed': "WARNING: Position disappeared without closure",
                        'Gain$': 0.0,
                        'Gain%': 0.0
                    }
                    
                    # Log the recovered trade
                    trade_to_log = {
                        'Ticker': trade['ticker'],
                        'Strike': trade['strike'],
                        'ExpDate': trade['expiration_date'],
                        'DayIn': trade['entry_date'],
                        'PriceIn': trade['premium_received'],
                        'Qty': trade['quantity'],
                        'AmountIn': trade['premium_received'] * trade['quantity'] * 100.0,
                        'unique_key': trade['unique_key'], # Add unique_key for validation
                        **exit_details
                    }
                    closed_trades_log.append(trade_to_log)
                    # Update counters for recovered close (Gain$ is 0.0)
                    closed_trades_count += 1
                    total_exit_events += 1  # Count recovered exits
                    
                    # Update tracker
                    # NOTE: open_puts_tracker counts open POSITIONS (one per entry),
                    # while 'quantity' is number of contracts per position.
                    # Previously we subtracted `trade['quantity']` here which mixed
                    # the two semantics and could incorrectly reduce the printed
                    # per-ticker counts (making positions appear to 'disappear').
                    prev_count = open_puts_tracker.get(trade['ticker'], 0)
                    open_puts_tracker[trade['ticker']] = max(0, prev_count - 1)
                    # Debug log to surface unexpected large deltas
                    if prev_count - open_puts_tracker[trade['ticker']] > 1:
                        print(f"⚠️ **DEBUG:** Adjusted open_puts_tracker for {trade['ticker']} by {prev_count - open_puts_tracker[trade['ticker']]} (prev {prev_count} -> now {open_puts_tracker[trade['ticker']]})")
            
            # Update peak open positions after processing exits and recoveries
            if current_account_put_positions > peak_open_positions:
                peak_open_positions = current_account_put_positions

            # --- DIAGNOSTIC CHECK: Verify open_trades_log matches open_puts_tracker ---
            open_trades_count = len(open_trades_log)
            open_puts_tracker_sum = sum(open_puts_tracker.values())
            priceable_positions = len(daily_liability_itemization) if 'daily_liability_itemization' in locals() else 0
            
            if open_trades_count != open_puts_tracker_sum:
                print(f"⚠️ **[DIAGNOSTIC] Mismatch after reconciliation on {date_str}:**")
                print(f"    open_trades_log count: {open_trades_count}")
                print(f"    open_puts_tracker sum: {open_puts_tracker_sum}")
                print(f"    priceable positions (in liability): {priceable_positions}")
                # Print per-ticker details
                tracker_counts = {k: v for k, v in open_puts_tracker.items() if v > 0}
                print(f"    open_puts_tracker details: {tracker_counts}")
                # List unique_keys still open
                open_keys = [(t['ticker'], t['strike'], t['expiration_date']) for t in open_trades_log]
                print(f"    open_trades_log positions ({len(open_keys)}): {open_keys}")

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
                                            annual_risk = None
                                            expected_profit = None
                                            
                                            if pbidpx_value > 0 and strike_value > pbidpx_value:
                                                # Base R/R ratio
                                                risk_reward_ratio = calculate_risk_reward_ratio(strike_value, pbidpx_value)
                                                # Compute annualized R/R (if DTE available)
                                                try:
                                                    if risk_reward_ratio is not None and isinstance(dte, int) and dte > 0:                                                                                                                
                                                        annual_risk = risk_reward_ratio * (365.0 / float(dte))
                                                except Exception:
                                                    annual_risk = None

                                                # Compute expected profit metric when Delta is available
                                                try:
                                                    if put_delta_value is not None:
                                                        # put_delta_value is a decimal (e.g., -0.3)
                                                        expected_profit = (pbidpx_value * (1.0 + put_delta_value) + (strike_value - pbidpx_value) * put_delta_value) / pbidpx_value
                                                except Exception:
                                                    expected_profit = None

                                                # Apply ALL three filters as minimum requirements
                                                passes_rr = risk_reward_ratio is not None and risk_reward_ratio > MIN_RISK_REWARD_RATIO
                                                passes_annual = annual_risk is not None and annual_risk > MIN_ANNUAL_RISK
                                                passes_expected = expected_profit is not None and expected_profit > MIN_EXPECTED_PROFIT
                                                
                                                passes_metric = passes_rr and passes_annual and passes_expected
                                            
                                            if passes_metric:
                                                # Store computed metrics and Adj. Close on the option for easy access
                                                option['calculated_rr_ratio'] = risk_reward_ratio
                                                option['annual_risk'] = annual_risk
                                                option['expected_profit'] = expected_profit
                                                option['adj_close'] = current_adj_close 
                                                filtered_options.append(option)
                                                
                                        # Only record the chain if it has at least one option that passed all filters
                                        if filtered_options:
                                            # Sort the options within the chain by the selected ranking metric
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


                
                
                # --- DAILY TRADE ENTRY LOOP (up to MAX_PUTS_PER_DAY new positions) ---
                while daily_entries_count < MAX_PUTS_PER_DAY:
                    
                    # --- Select the ABSOLUTE BEST NON-DUPLICATE, LIMIT-RESPECTING Contract of the Day ---
                    
                    best_contract = None
                    trade_quantity = 0
                    ask_at_entry_float = 0.0 # FIX: Renamed variable to reflect float status
                    bid_at_entry = 0.0

                    if daily_trade_candidates:
                        # Sort the ENTIRE list of candidates globally by the selected ranking metric
                        if RANK_BY_ANNUAL:
                            sort_key = lambda x: x.get('annual_risk', -float('inf'))
                        elif RANK_BY_EXPECTED:
                            sort_key = lambda x: x.get('expected_profit', -float('inf'))
                        else:
                            # Default: risk/reward ratio (also used when RANK_BY_RR is True)
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

                            # Entry number for today (1, 2, 3, etc.)
                            entry_number_today = daily_entries_count + 1
                            
                            # Safely extract and convert all numeric fields
                            ticker = best_contract['ticker']
                            strike = float(best_contract['strike'])
                            dte = int(best_contract['dte'])
                            rr_ratio = float(best_contract['calculated_rr_ratio'])
                            bid_px = float(best_contract['pBidPx'])
                            expiration_date = best_contract['expiration_date']
                            
                            # Extract delta
                            delta_raw = best_contract.get('putDelta')
                            delta_val = safe_percentage_to_float(delta_raw) if isinstance(delta_raw, str) else delta_raw
                            delta_display = delta_val if delta_val is not None else 0.0
                            
                            exp_profit_raw = best_contract.get('expected_profit', 0.0)
                            exp_profit_val = safe_percentage_to_float(exp_profit_raw) if isinstance(exp_profit_raw, str) else exp_profit_raw
                            exp_profit_pct = (exp_profit_val * 100) if exp_profit_val is not None else 0.0
                            
                            best_info = (
                                f"  {entry_number_today}. **{ticker}:** Qty={trade_quantity}, "
                                f"Total Premium Collected=${total_premium_collected:,.2f}, "
                                f"Bid=${bid_px:.2f}, "
                                f"Strike=${strike:.2f}, "
                                f"DTE={dte}, "
                                f"Expiration Date={expiration_date}, "
                                f"Delta={delta_display:.4f}, "
                                f"R/R={rr_ratio:.2f}, "       
                                f"Annual Risk={best_contract['annual_risk']:.2f}, "                      
                                f"ExpProfit={exp_profit_pct:.2f}%, "
                                f"AdjClose=${adj_close:.2f}, "                                
                                f"Strike/AdjClose Ratio={strike_adj_close_ratio:.2f}%"
                            )
                            print(best_info)
                            
                        else:
                            print("❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (All candidates failed limits/duplication checks or resulted in Qty=0).")
                            
                    else:
                        print(f"❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (No contract passed filters).")
                        # Print portfolio summary on days with no viable contracts
                        total_positions = sum(open_puts_tracker.values())
                        print(f"\n**OPEN PORTFOLIO SUMMARY ({total_positions} Total Positions):**")
                        if total_positions > 0:                            
                            print_daily_portfolio_summary(open_puts_tracker)
                    
                    
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
                        # Update current and peak open positions after entry
                        current_account_put_positions = sum(open_puts_tracker.values())
                        if current_account_put_positions > peak_open_positions:
                            peak_open_positions = current_account_put_positions
                        
                        # 6. Log the trade details (include quantity)
                        # Determine the entry stock AdjClose for this ticker on entry day
                        try:
                            entry_adj_close_value = best_contract.get('adj_close', None)
                        except Exception:
                            entry_adj_close_value = None
                        if entry_adj_close_value is None:
                            try:
                                entry_adj_close_value = stock_history_dict.get(ticker_to_enter, {}).get(date_str, {}).get('adj_close')
                            except Exception:
                                entry_adj_close_value = None

                        trade_entry = {
                            'entry_date': daily_date_obj.strftime('%Y-%m-%d'),
                            'ticker': ticker_to_enter,
                            'strike': best_contract['strike'],
                            'expiration_date': best_contract['expiration_date'],
                            'premium_received': bid_at_entry, 
                            'quantity': trade_quantity,
                            'entry_adj_close': entry_adj_close_value,
                            'unique_key': (ticker_to_enter, best_contract['strike'], best_contract['expiration_date']),
                            'last_known_ask': ask_at_entry_float,  # Store initial ask price
                            'last_ask_date': date_str  # Store date of last known ask price
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
                        
                        # Increment daily entry counter and remove entered contract from candidates
                        daily_entries_count += 1
                        
                        # Remove the entered contract from the candidates list to prevent re-selection
                        daily_trade_candidates.remove(best_contract)
                        
                    else:
                        # No valid contract found in this iteration, exit the daily entry loop
                        break
                
                # --- END OF DAILY TRADE ENTRY LOOP ---
            
            # ----------------------------------------------
            # --- FINAL EOD VALUATION CALCULATIONS ---
            # --- THIS BLOCK MUST RUN AFTER ALL ENTRIES/EXITS ARE COMPLETE ---
            # ----------------------------------------------
            
            # Re-initialize liability trackers to correctly sum up current state
            unrealized_pnl = 0.0 
            total_put_liability = 0.0 
            total_open_premium_collected = 0.0 
            daily_liability_itemization = []
            unpriceable_positions = []  # Track positions that can't be priced

            for trade in open_trades_log:
                # Use the conservative exit price (Ask price) for valuation
                current_price = get_contract_exit_price(
                    daily_orats_data, 
                    trade['ticker'], 
                    trade['expiration_date'], 
                    trade['strike']
                )
                
                # Update last_known_ask if we got a valid price today
                if current_price is not None:
                    trade['last_known_ask'] = current_price
                    trade['last_ask_date'] = date_str
                    price_source_date = date_str
                else:
                    # Use the stored last known ask price
                    current_price = trade.get('last_known_ask')
                    price_source_date = trade.get('last_ask_date', 'unknown')
                
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
                    if price_source_date != date_str:
                        # Using historical price - note it in the itemization
                        item_detail = (
                            f"  > **{trade['ticker']}** (Qty {trade['quantity']}, Strike ${trade['strike']:.2f}, Exp {trade['expiration_date']}): "
                            f"Ask=${current_price:.2f} (from {price_source_date}), Cost to Close=${put_cost_to_close:,.2f}"
                        )
                    else:
                        item_detail = (
                            f"  > **{trade['ticker']}** (Qty {trade['quantity']}, Strike ${trade['strike']:.2f}, Exp {trade['expiration_date']}): "
                            f"Ask=${current_price:.2f}, Cost to Close=${put_cost_to_close:,.2f}"
                        )
                    daily_liability_itemization.append(item_detail)
                    
                else:
                    # CRITICAL: Track unpriceable positions for diagnostic reporting
                    # This should rarely happen now since we store last_known_ask
                    unpriceable_positions.append({
                        'ticker': trade['ticker'],
                        'strike': trade['strike'],
                        'expiration': trade['expiration_date'],
                        'quantity': trade['quantity']
                    })
                    # Always log unpriceable contracts (not just in DEBUG_VERBOSE mode)
                    print(f"⚠️ **UNPRICEABLE POSITION (No ask price ever recorded):** {trade['ticker']} Strike ${trade['strike']:.2f}, Exp {trade['expiration_date']}, Qty {trade['quantity']} - Position still open but excluded from today's liability/NAV calculation")

            # Report unpriceable positions summary if any exist
            if unpriceable_positions:
                print(f"⚠️ **WARNING: {len(unpriceable_positions)} position(s) could not be priced today and are excluded from liability calculation**")
                print(f"    IMPACT: Today's Total Account Value (NAV) and Unrealized P&L calculations may be inaccurate.")
                print(f"    REASON: ORATS data file for {date_str} is missing ask prices for these contracts.")

            # --- DIAGNOSTIC CHECK: Verify position counts match ---
            open_trades_count = len(open_trades_log)
            priceable_count = len(daily_liability_itemization)
            unpriceable_count = len(unpriceable_positions)
            open_puts_tracker_sum = sum(open_puts_tracker.values())  # Yuda
            print(f"  Open Puts: {open_puts_tracker_sum:,.2f}")
            
            # The sum should equal: priceable + unpriceable = open_trades_count = open_puts_tracker_sum
            if priceable_count + unpriceable_count != open_trades_count:
                print(f"⚠️ **[DIAGNOSTIC] Position count mismatch on {date_str}:**")
                print(f"    open_trades_log: {open_trades_count}")
                print(f"    priceable (in liability): {priceable_count}")
                print(f"    unpriceable (no ask): {unpriceable_count}")
                print(f"    priceable + unpriceable: {priceable_count + unpriceable_count}")
                print(f"    EXPECTED: All three should equal {open_trades_count}")
            
            if open_trades_count != open_puts_tracker_sum:
                print(f"⚠️ **[DIAGNOSTIC] Tracker mismatch on {date_str}:**")
                print(f"    open_trades_log: {open_trades_count}")
                print(f"    open_puts_tracker sum: {open_puts_tracker_sum}")
                tracker_counts = {k: v for k, v in open_puts_tracker.items() if v > 0}
                print(f"    open_puts_tracker: {tracker_counts}")

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
            
            # Prepare monthly summary message every day (efficient: no lookahead needed)
            elapsed_seconds = int(time.perf_counter() - _sim_start_time)
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            seconds = elapsed_seconds % 60
            runtime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            monthly_summary_msg = f"📅 {daily_date_obj}   Account Value: ${total_account_value:12,.2f}   RunTime: {runtime_str}"
            
            # Print the prepared message when month changes (i.e., on first day of new month, print last month's final values)
            if last_printed_month is not None and last_printed_month != month_key:
                # Month changed - print the last message we prepared (from previous month's last day)
                if hasattr(sys.stdout, 'force_print'):
                    sys.stdout.force_print(last_monthly_summary_msg)
                else:
                    print(last_monthly_summary_msg)
            
            # Store this message for potential printing when month changes
            last_monthly_summary_msg = monthly_summary_msg
            last_printed_month = month_key
            
            # Print Account Value breakdown (Corrected for Accuracy and Transparency)
            print(f"💵 **DAILY ACCOUNT VALUE (EOD - NAV):** ${total_account_value:,.2f}")            
            print(f"  > **Cash Balance:** ${cash_balance:,.2f}")
            print(f" SPY current price: {spy_current_price}")
            open_puts_tracker_sum = sum(open_puts_tracker.values())  # Yuda
            print(f"  Open Puts: {open_puts_tracker_sum:,.2f}")
            # --- PROMOTED LIABILITY PRINT (This is the cumulative value) ---
            print(f"🛑 **TOTAL PORTFOLIO LIABILITY (Cost to Close):** ${total_put_liability:,.2f} (Computed using Ask Price)") # Yuda
            
            # Print Itemized Liability Breakdown
            if daily_liability_itemization:
                for item in daily_liability_itemization:
                    print(item)

            
            print(f"  > **Total accumulated Premium on Open Puts:** +${total_open_premium_collected:,.2f}")

            print(f"💰 **Cash Balance:** ${cash_balance:,.2f}")

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

            # Net Unrealized P&L is still calculated using the old definition: (Premium - Liability). 
            # We display it here for informational purposes, but it is NOT used in NAV.
            print(f"  > **Net Unrealized P&L:** ${unrealized_pnl:,.2f}")

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

            # Print cumulative realized P&L (Total net realized P&L)
            print(f"💰 **TOTAL NET REALIZED P&L (Cumulative):** ${cumulative_realized_pnl:,.2f}")

            # Print total P&L (Realized + Unrealized)
            print(f"💰 **TOTAL P&L (Realized + Unrealized):** ${(cumulative_realized_pnl + unrealized_pnl):,.2f}")

            # Print cash basis + total P&L (helpful sanity check: cash + (realized+unrealized))

            # Compute from current INITIAL_CASH + total P&L
            cash_plus_total_pnl = INITIAL_CASH + (cumulative_realized_pnl + unrealized_pnl)
            print(f"🧾 **INITIAL_CASH + TOTAL P&L (Cash Basis):** ${cash_plus_total_pnl:,.2f}")
            print(f"💵 **DAILY ACCOUNT VALUE (EOD - NAV):** ${total_account_value:,.2f}")   
            print(f" SPY current price: {spy_current_price}")

            # Compare with NAV (total_account_value). If mismatch, print and quit.
            try:
                # Allow a tiny numerical tolerance (1 cent)
                if abs(cash_plus_total_pnl - float(total_account_value)) <= 0.01:
                    print("✅ Total is the same as cash+gain")
                else:
                    print("❌ Total is not the same as cash+gain")
                    print(f"  | cash_plus_total_pnl: ${cash_plus_total_pnl:,.2f}")
                    print(f"  | total_account_value: ${total_account_value:,.2f}")
                    # sys.exit(1)
            except Exception:
                # If comparison or exit fails for any reason, continue but report
                print("⚠️ Could not compare cash+gain to total_account_value due to an internal error.")

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
    
    # Print the final month's summary (last month we processed)
    if last_monthly_summary_msg is not None:
        if hasattr(sys.stdout, 'force_print'):
            sys.stdout.force_print(last_monthly_summary_msg)
        else:
            print(last_monthly_summary_msg)
    
    # 6. Final Liquidation and Performance Summary (Runs after all trading days are processed)
    
    # Enable output if minimal mode was active
    if hasattr(sys.stdout, 'enable_output'):
        sys.stdout.enable_output()
    
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

            # Fallbacks: use stored last_known_ask, then force $0.00 as a last resort
            fallback_note = ""
            if closing_ask is None:
                stored_price = trade.get('last_known_ask')
                stored_date = trade.get('last_ask_date', 'unknown')
                if stored_price is not None:
                    closing_ask = stored_price
                    fallback_note = f" (Price from {stored_date})"
                else:
                    closing_ask = 0.0
                    fallback_note = " (No price data; forced at $0.00)"
            
            qty = trade['quantity']
            
            premium_collected_per_contract = trade['premium_received']
            
            # Financials (based on last known/available market price)
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
            
            reason_str = "Last day liquidation" + (fallback_note if fallback_note else "")
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
                'ReasonWhyClosed': reason_str,
                'Gain$': position_net_gain,
                'Gain%': position_gain_percent,
                'unique_key': trade['unique_key'], # Add unique_key for validation
            }
            closed_trades_log.append(trade_to_log)
            # Update win/loss counters for liquidation closes
            closed_trades_count += 1
            if position_net_gain > 0:
                winning_trades_count += 1
            
            use_older_ask = " using older ask price" if fallback_note else ""
            print(
                f"| {trade['ticker']:<6} | {qty:3} | $ {trade['strike']:>6.2f} | $ {premium_collected_gross:>11,.2f} | "
                f"$ {closing_ask:>10.2f} | $ {cost_to_close_gross:>12,.2f} | $ {exit_commission:>13.2f} | "
                f"$ {position_net_gain:>11.2f} | {use_older_ask}"
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
    print("| Month   | Total Value EOD    | $ Gain            |  % Gain | % SPY Gain |")
    print("|---------|--------------------|-------------------|---------|------------|") 
    
    for (year, month), data in monthly_performance.items():
        month_label = datetime(year, month, 1).strftime('%Y-%m')
        
        # Data widths used: End Value (11,.2f), $ Gain (9,.2f), % Gain (6.2f), % SPY Gain (8.2f)
        print(
            f"| {month_label:^5} | $ {data['end_value']:>14,.2f}   | $ {data['gain_abs']:>15,.2f} | "
            f"{data['gain_pct']:>6.2f}% | {data['spy_gain_pct']:>9.2f}% |"
        )

    # Cumulative monthly $ Gain total
    try:
        total_monthly_gain_abs = sum(d.get('gain_abs', 0.0) for d in monthly_performance.values())
        print("|---------|--------------------|-------------------|---------|------------|")
        print(f"| {'TOTAL (Months)':<9} {' ':13} | $ {total_monthly_gain_abs:>15,.2f} |")
    except Exception:
        pass

    # --- Print Yearly Table ---
    print("")
    print("\n--- YEARLY PORTFOLIO GAIN ---")
    # NEW COLUMN: % SPY Gain
    print("| Year    | Total Value EOD    | $ Gain               | % Gain  | % SPY Gain |")
    print("|---------|--------------------|----------------------|---------|------------|") 
    
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
            f"| {year:^5}   | $ {year_end_value:>14,.2f}   | $ {yearly_gain_abs:>18,.2f} | "
            f"{yearly_gain_pct:>6.2f}% | {spy_yearly_return:>9.2f}% |"
        )

    # Cumulative yearly $ Gain total
    try:
        total_yearly_gain_abs = sum((data.get('end_value', 0.0) - data.get('start_value', 0.0)) for data in yearly_performance.values())        
        print(f"|---------|--------------------|----------------------|---------|------------|")
        print(f"| {'TOTAL (Years)':<9} {' ':14} | $ {total_yearly_gain_abs:>18,.2f} | ")
    except Exception:
        pass

    # 9. Exit Statistics (Focusing on Entry/Exit Events)
    total_closed_positions_qty = stop_loss_count + expired_otm_count + expired_itm_count
    
    # Calculate Total Exit Events (Total number of discrete trades closed)
    total_exit_events_count = len(closed_trades_log)

    print("")
    print("\n--- TRADE EXIT STATISTICS (by Trade Event Count) ---")
    
    # Define Total Gain/Premium Collected for the whole simulation
    TOTAL_GAIN = stop_loss_gain + take_profit_gain + expired_otm_gain + expired_itm_gain + liquidation_gain
    TOTAL_PREMIUM_COLLECTED = stop_loss_premium_collected + take_profit_premium_collected + expired_otm_premium_collected + expired_itm_premium_collected + liquidation_premium_collected
    
    # --- Build detailed exit reason breakdown ---
    # Normalize stop-loss reasons into the 4 canonical categories from exit_put_position
    def _normalize_reason(reason_raw: str) -> str:
        if not reason_raw:
            return 'Unknown'
        r = str(reason_raw).strip()
        rl = r.lower()
        # Take Profit
        if 'takeprofit' in rl:
            return 'TakeProfit Min gain reached'
        # Expiration
        if 'expiration' in rl and 'itm' in rl:
            return 'Expiration (ITM/Assigned)'
        if 'expiration' in rl and 'otm' in rl:
            return 'Expiration (OTM/Max Profit)'
        # Liquidation
        if 'liquidation' in rl or 'last day' in rl:
            return 'Last day liquidation'
        # Stop-Loss canonicalization (4 reasons)
        if 'stoploss' in rl:
            if 'bid' in rl:
                return 'StopLoss: Option Bid Above Entry'
            if 'strike' in rl:
                return 'StopLoss: Stock Below Strike Threshold'
            if 'entry' in rl:
                return 'StopLoss: Stock Below Entry Threshold'
            if 'sma150' in rl or 'avg' in rl:
                return 'StopLoss: Stock Below SMA150'
            return 'StopLoss: Other'
        return r

    # Count events and sum gains per specific (normalized) exit reason
    exit_reason_stats = {}
    for trade in closed_trades_log:
        raw_reason = trade.get('ReasonWhyClosed', 'Unknown')
        reason = _normalize_reason(raw_reason)
        if reason not in exit_reason_stats:
            exit_reason_stats[reason] = {'count': 0, 'gain': 0.0}
        exit_reason_stats[reason]['count'] += 1
        exit_reason_stats[reason]['gain'] += trade.get('Gain$', 0.0)
    
    # Ensure ITM expiration row is always shown, even if zero events occurred
    if 'Expiration (ITM/Assigned)' not in exit_reason_stats:
        exit_reason_stats['Expiration (ITM/Assigned)'] = {'count': 0, 'gain': 0.0}
    
    # Total Closed Trades
    total_closed_events = len(closed_trades_log)

    # Calculate Net % Gain relative to premium collected for each category
    def calculate_net_gain_percent(gain, premium):
        # We calculate P&L / Premium Collected (Return on Premium)
        return (gain / premium) * 100.0 if premium != 0.0 else 0.0
    
    # Sort reasons alphabetically for consistent display
    sorted_reasons = sorted(exit_reason_stats.keys())
    
    # Header
    print("| Exit Reason                          | Exit Events  |  % of Total | Total Gain $      | Net Gain % |")
    print("|--------------------------------------|--------------|-------------|-------------------|------------|")
    
    # Print each specific exit reason
    for reason in sorted_reasons:
        stats = exit_reason_stats[reason]
        count = stats['count']
        gain = stats['gain']
        pct_of_total = (count / total_closed_events * 100) if total_closed_events > 0 else 0
        
        # Determine which premium bucket this reason belongs to for Net Gain %
        if 'Expiration (ITM' in reason:
            net_gain_pct = calculate_net_gain_percent(gain, expired_itm_premium_collected) if expired_itm_premium_collected > 0 else 0
        elif 'Expiration (OTM' in reason:
            net_gain_pct = calculate_net_gain_percent(gain, expired_otm_premium_collected) if expired_otm_premium_collected > 0 else 0
        elif 'LIQUIDATION' in reason:
            net_gain_pct = calculate_net_gain_percent(gain, liquidation_premium_collected) if liquidation_premium_collected > 0 else 0
        elif 'TakeProfit' in reason:
            # For take-profit rows, use take-profit premium as baseline
            net_gain_pct = calculate_net_gain_percent(gain, take_profit_premium_collected) if take_profit_premium_collected > 0 else 0
        elif 'StopLoss' in reason:
            # For stop-loss rows, use stop-loss premium as baseline
            net_gain_pct = calculate_net_gain_percent(gain, stop_loss_premium_collected) if stop_loss_premium_collected > 0 else 0
        else:
            net_gain_pct = 0
        
        # Truncate reason if too long (max 36 chars to fit column)
        reason_display = reason[:36] if len(reason) <= 36 else reason[:33] + "..."
        
        print(
            f"| {reason_display:<36} | {count:>12,} | {pct_of_total:>10.2f}% | "
            f"${gain:>16,.2f} | {net_gain_pct:>9.2f}% |"
        )
    
    # Separator and Total
    print(f"|--------------------------------------|--------------|-------------|-------------------|------------|")    
    print(f"| Total Exit Trades Closed             | {total_closed_events:>12,} | {100.0:>10.2f}% | "
          f"${TOTAL_GAIN:>16,.2f} | {'N/A':>10} |"
    )
    print(f"|--------------------------------------|--------------|-------------|-------------------|------------|")     
    print(f"| Total Entry Events{' ':19}| {total_entry_events:>12,} |")
    
    
    # 10. NEW: Detailed Closed Trade Log
    if closed_trades_log:
        
        # Sort the log by exit date
        closed_trades_log.sort(key=lambda x: x['DayOut'])       

        # Adjusted separator for new Exit # column
        print("\n\n--- DETAILED CLOSED TRADE LOG (Full History) ---")
        print("| Exit #  | Ticker |  Qty |   Day In   | Price In   | Amount In   |  Day Out   | Price Out |  Amount Out | Reason Why Closed          |    Gain $  |   Gain % |")
        print("|---------|--------|------|------------|------------|-------------|------------|-----------|-------------|----------------------------|------------|----------|")
        
        for index, trade in enumerate(closed_trades_log):
            
            # Exit Number (1, 2, 3...)
            exit_number = index + 1
            
            # Format numbers (Price In/Out, Amount In/Out, Gain $)
            price_in_str = f"${trade['PriceIn']:>9.2f}"
            price_out_str = f"${trade['PriceOut']:>8.2f}" if trade['PriceOut'] is not None else ""
            
            amount_in_str = f"${trade['AmountIn']:>10,.2f}"
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

        # Cumulative totals for closed trades (AmountIn, AmountOut, Gain$)
        try:
            total_amount_in = sum(float(t.get('AmountIn') or 0.0) for t in closed_trades_log)
            total_amount_out = sum(float(t.get('AmountOut') or 0.0) for t in closed_trades_log)
            total_gain_dollars = sum(float(t.get('Gain$') or 0.0) for t in closed_trades_log)
            print()
            print("--- CLOSED TRADES CUMULATIVE TOTALS ---")
            print(f"  Total Amount In : ${total_amount_in:,.2f}")
            print(f"  Total Amount Out: ${total_amount_out:,.2f}")
            print(f"  Total Net Gain $ : ${total_gain_dollars:,.2f}")
        except Exception:
            # If any unexpected format occurs, skip totals but continue gracefully
            pass

    print("\n=== FINAL TRADING RULES SUMMARY ===\n")
    
    # 1. Account Simulation Rules
    print(f"📊 Account Simulation Rules")
    print(f"|----------------------------|----------------|")
    print(f"| Parameter                  | Value          |")
    print(f"|----------------------------|----------------|")
    print(f"| Start Date                 | {start_date_str:<14} |")
    print(f"| Initial Cash               | ${float(rules['account_simulation']['initial_cash']):>13,.2f} |")
    print(f"| Max Puts/Account           | {MAX_PUTS_PER_ACCOUNT:>14} |")
    print(f"| Max Puts/Stock             | {MAX_PUTS_PER_STOCK:>14} |")
    print(f"| Max Puts/Day               | {MAX_PUTS_PER_DAY:>14} |")
    print(f"|----------------------------|----------------|")
    print()   

    # 2. Underlying Stock Rules (Final Summary)
    print(f"🧩 Underlying Stock Rules")
    print(f"|----------------------------|----------------|")
    print(f"| Parameter                  | Value          |")
    print(f"|----------------------------|----------------|")
    print(f"| Min 5-Day Rise             | {min_rise_str} |")
    print(f"| Min Above Avg              | {min_above_str} |")
    print(f"| Max Above Avg              | {max_above_str} |")
    print(f"| Min 10-Day Avg Slope       | {min_slope_str} |")
    print(f"| Min Stock Price            | {min_price_str} |")
    print("|----------------------------|----------------|")
    print()

    # 3. Entry Put Position Rules
    print("📈 Entry Put Position Rules")
    print(f"|----------------------------|----------------|")
    print(f"| Parameter                  | Value          |")
    print(f"|----------------------------|----------------|")
    print(f"| Min DTE                    | {MIN_DTE:>14} |")
    print(f"| Max DTE                    | {MAX_DTE:>14} |")
    print(f"| Min Put Bid Price          | $ {MIN_BID_PRICE:>12.2f} |")
    print(f"| Min Put Delta              | {MIN_DELTA*100:>13.1f}% |")
    print(f"| Max Put Delta              | {MAX_DELTA*100:>13.1f}% |")
    print(f"| Max Bid-Ask Spread         | {MAX_SPREAD_DECIMAL*100:>13.1f}% |")
    print(f"| Min Avg Above Strike       | {MIN_AVG_ABOVE_STRIKE_PCT*100:>13.1f}% |")
    print(f"| Min Risk/Reward Ratio      | {MIN_RISK_REWARD_RATIO:>14.1f} |")    
    print(f"| Min Annual Risk            | {MIN_ANNUAL_RISK:>14.1f} |")
    print(f"| Min Expected Profit        | {MIN_EXPECTED_PROFIT*100:>13.1f}% |")
    print(f"| Rank By Risk/Reward Ratio  | {('Yes' if RANK_BY_RR else 'No'):>14} |")
    print(f"| Rank By Annual Risk        | {('Yes' if RANK_BY_ANNUAL else 'No'):>14} |")
    print(f"| Rank By Expected Profit    | {('Yes' if RANK_BY_EXPECTED else 'No'):>14} |")
    print(f"|----------------------------|----------------|")
    print()

    # 4. Exit Put Position Rules
    print("📉 Exit Put Position Rules")
    print(f"|----------------------------|----------------|")
    print(f"| Parameter                  | Value          |")
    print(f"|----------------------------|----------------|")
    print(f"| Position Stop Loss         | {POSITION_STOP_LOSS_PCT*100:>13.1f}% |")
    print(f"| Stock Below SMA150         | {STOCK_MAX_BELOW_AVG_PCT*100:>13.1f}% |")
    print(f"| Stock Min Above Strike     | {STOCK_MIN_ABOVE_STRIKE_PCT*100:>13.1f}% |")   
    print(f"| Stock Max Below Entry      | {STOCK_MAX_BELOW_ENTRY_PCT*100:>13.1f}% |") 
    print(f"| Min Gain to Take Profit    | {TAKE_PROFIT_MIN_GAIN_PCT*100:>13.1f}% |")
    print(f"|----------------------------|----------------|")
    print()   

    # 4a. Trading Costs and Limits 
    print("💰 Trading Parameters")
    print(f"|----------------------------|----------------|")
    print(f"| Parameter                  | Value          |")
    print(f"|----------------------------|----------------|")
    print(f"| Commission/Contract        | ${COMMISSION_PER_CONTRACT:>13.2f} |")
    print(f"| Max Premium/Trade          | ${MAX_PREMIUM_PER_TRADE:>13.2f} |")
    print(f"|----------------------------|----------------|")
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

    # Calculate Win Ratio (using incremental counters)
    win_ratio_pct = (winning_trades_count / closed_trades_count * 100.0) if closed_trades_count > 0 else 0.0
    
    # Performance Summary
    print("📊 Final Performance")
    print(f"|----------------------------|----------------------|") 
    print(f"| Parameter                  |  Value               |")
    print(f"|----------------------------|----------------------|")
    print(f"| Current Date/Time          | {datetime.now().strftime('%Y-%m-%d %H:%M'):>20} |")
    print(f"| Annualized Gain            | {annualized_gain:>19.2f}% |")
    print(f"| Total Gain                 | ${TOTAL_GAIN:>19,.2f} |")    
    print(f"| Run Time                   | {runtime_str:>20} |")
    print(f"| Peak Open Positions        | {peak_open_positions:>20} |")
    print(f"| Total Entry Events         | {total_entry_events:>20} |")
    print(f"| Win Ratio                  | {win_ratio_pct:>19.2f}% |")
    
    # Print current log file name (row ~2418 request)
    try:
        current_log_path = getattr(sys.stdout, 'logfile', None)
        if current_log_path and hasattr(current_log_path, 'name'):
            log_filename_only = os.path.basename(current_log_path.name)
            print(f"| Log File                   | {log_filename_only:>20} |")
        else:
            # Fallback if stdout has been restored or structure changed
            print(f"| Log File                   | {'N/A':>20} |")
    except Exception:
        print(f"| Log File                   | {'ERR':>20} |")
    
    # Worst drawdown across all simulated dates
    try:
        print(f"| Worst Drawdown             | {worst_drawdown_pct:>19.2f}% |")
    except Exception:
        # If for any reason the metric isn't available, skip gracefully
        pass
    print(f"|----------------------------|----------------------|")
    print()    
    
# Execute the main function
if __name__ == "__main__":
    load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)