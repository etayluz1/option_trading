# Utility: Find first dividend and split event between two dates for a ticker
def get_first_dividend_and_split(stock_history_dict, ticker, entry_date, exit_date):
    """
    Returns (dividend_str, split_date_str): the date of the first dividend and split event (as yyyy-mm-dd), or '' if none,
    between entry_date and exit_date (inclusive) for the given ticker.
    """
    if ticker not in stock_history_dict:
        return '', ''
    ticker_data = stock_history_dict[ticker]
    # Ensure dates are in correct order and format
    try:
        entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
        exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
    except Exception:
        return '', ''
    if entry_dt > exit_dt:
        entry_dt, exit_dt = exit_dt, entry_dt
    dividend_str = ''
    split_date_str = ''
    for dt in sorted(ticker_data.keys()):
        try:
            dt_obj = datetime.strptime(dt, '%Y-%m-%d')
        except Exception:
            continue
        if entry_dt <= dt_obj <= exit_dt:
            day_data = ticker_data[dt]
            # Check for dividend
            if dividend_str == '' and (day_data.get('Dividends', 0) or day_data.get('dividends', 0)):
                dividend_str = dt
            # Check for split
            split_val = day_data.get('Split', 0) or day_data.get('Split_Ratio', 0) or day_data.get('split', 0)
            if split_date_str == '' and split_val:
                split_date_str = dt
            if dividend_str and split_date_str:
                break
    return dividend_str, split_date_str
import orjson
import os
from datetime import datetime
import math
import sys
import time

# ----- Constants for ORATS.json option data indices -----
STRIKE_ID = 0
BID_ID = 1
ASK_ID = 2
DELTA_ID = 3

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
import os
# Allow override of stock_history.json path via environment variable for wrapper use
JSON_FILE_PATH = os.environ.get("SIM_WRAPPER_STOCK_HISTORY_PATH", "stock_history.json")
TARGET_TICKER = "SPY" # Retained for context, but the script processes ALL tickers.
DEBUG_VERBOSE = False # Set to True to see individual ticker details (Total Viable Options / Details by DTE)

# Commission Fee
COMMISSION_PER_CONTRACT = 0.67
commission_per_contract_str = f"{COMMISSION_PER_CONTRACT:>13.4f}"
FINAL_COMMISSION_PER_CONTRACT = COMMISSION_PER_CONTRACT # Commission for closing trades

# Maximum premium to collect per single trade entry
MAX_PREMIUM_PER_TRADE = 5000.00 

# Global mode flag: start in low puts mode (True). When False, simulation runs in high-puts mode.
# True means start in low mode (fewer puts). The simulation may flip this flag at runtime
# based on configured thresholds (e.g. low_min_puts_to_set_low_mode / low_max_puts_to_set_high_mode).
low_puts_mode = True

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
        try:
            option_strike = float(option[STRIKE_ID])
            pbidpx = float(option[BID_ID])
            paskpx = float(option[ASK_ID])
            # Convert delta back to negative percentage string for compatibility
            put_delta = -abs(float(option[DELTA_ID]))
            put_delta_str = f"{put_delta:.4f}%"
        except (IndexError, ValueError, TypeError):
            continue
        if abs(option_strike - strike) < 0.001:
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
        try:
            option_strike = float(option[STRIKE_ID])
            pbidpx = float(option[BID_ID])
            # Convert delta back to negative percentage string for compatibility
            put_delta = -abs(float(option[DELTA_ID]))
            put_delta_str = f"{put_delta:.4f}%"
        except (IndexError, ValueError, TypeError):
            continue
        if abs(option_strike - strike) < 0.001:
            return pbidpx if pbidpx > 0 else None

    return None


def load_and_run_simulation(rules_file_path, json_file_path):
    """
    Loads rules and data, initializes the tracker, and iterates chronologically 
    over ALL daily entries for ALL tickers starting from the specified date.
    Implements exit logic, calculates daily P&L and account value, respects 
    portfolio limits, and sells the optimal quantity based on premium collected.
    """
    print(f"Start Simulation: Loading rules from '{rules_file_path}' and data from '{json_file_path}'")

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

    # Check if running from wrapper (has SIM_WRAPPER_LOG_FILENAME environment variable)
    log_filename = os.environ.get("SIM_WRAPPER_LOG_FILENAME")
    run_id = os.environ.get("SIM_WRAPPER_RUN_ID")  # Legacy support
    
    if log_filename is not None:
        # Use the filename provided by wrapper
        base_name = log_filename
    else:
        # Timestamped log filename: yyyy-mm-dd hh-mm.log (Windows-safe: ':' not allowed)
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
        if run_id is not None:
            base_name = f"{timestamp}_Run{run_id}.log"
        else:
            base_name = f"{timestamp}.log"
    
    log_file_path = os.path.join(LOG_DIR, base_name)
    
    # Ensure uniqueness if multiple runs occur within the same minute (only for non-wrapper runs)
    if log_filename is None:
        suffix = 1
        while os.path.exists(log_file_path):
            suffix += 1
            if run_id is not None:
                log_file_path = os.path.join(LOG_DIR, f"{timestamp}_{suffix}_Run{run_id}.log")
            else:
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
    global low_puts_mode  # Declare as global so we can modify it
    
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
            wrapper_sweep_pct_val = rules["account_simulation"].get("wrapper_sweep_pct", "5%")
            wrapper_sweep_pct_float = float(wrapper_sweep_pct_val.rstrip("%"))
            wrapper_sweep_pct_str = f"{wrapper_sweep_pct_float:>13.4f}%"
            drawdown_goal_pct_str = rules["account_simulation"].get("drawdown_goal_pct", "25%")
            drawdown_goal_pct = float(drawdown_goal_pct_str.strip('%'))
            MINIMAL_PRINT_OUT = bool(rules["account_simulation"].get("Minimal_Print_Out", False))

            # --- DRAWDOWN GOAL PCT ---
            
            # --- RISK MANAGEMENT RULE (Stock Price Stop Loss) ---
            STOCK_MAX_BELOW_AVG_PCT = abs(safe_percentage_to_float(rules["exit_put_position"]["stock_max_below_avg"]))
            stock_max_below_avg_str = f"{STOCK_MAX_BELOW_AVG_PCT*100:>13.4f}%"
            
            # Start/End Dates
            start_date_str = rules["account_simulation"]["start_date"]
            # Allow both mm/dd/yy and mm/dd/yyyy
            try:
                start_date_obj = datetime.strptime(start_date_str, '%m/%d/%y').date()
            except ValueError:
                try:
                    start_date_obj = datetime.strptime(start_date_str, '%m/%d/%Y').date()
                except ValueError as e:
                    raise ValueError(f"Invalid start_date format: '{start_date_str}'. Use mm/dd/yy or mm/dd/yyyy.") from e
            end_date_str = rules["account_simulation"].get("end_date")
            end_date_obj = None
            if end_date_str:
                # Support both 2-digit and 4-digit year formats
                try:
                    end_date_obj = datetime.strptime(end_date_str, '%m/%d/%y').date()
                except ValueError:
                    try:
                        end_date_obj = datetime.strptime(end_date_str, '%m/%d/%Y').date()
                    except ValueError:
                        end_date_obj = None
            
            # DTE Rules
            MIN_DTE_RULE = int(rules["entry_put_position"]["min_days_for_expiration"])
            MAX_DTE_RULE = int(rules["entry_put_position"]["max_days_for_expiration"])
            
            # Bid Price Rule
            MIN_BID_PRICE = float(rules["entry_put_position"]["min_put_bid_price"].replace('$', '').strip())
            min_bid_price_str = f"$ {MIN_BID_PRICE:>12.4f}"
                                  
            
            # Put Delta Rules
            MIN_DELTA = safe_percentage_to_float(rules["entry_put_position"]["min_put_delta"])
            MAX_DELTA = safe_percentage_to_float(rules["entry_put_position"]["max_put_delta"])
            min_delta_str = f"{MIN_DELTA*100:>13.4f}%"
            max_delta_str = f"{MAX_DELTA*100:>13.4f}%"           
            
            # Max Bid-Ask Spread Rule
            MAX_SPREAD_DECIMAL = safe_percentage_to_float(rules["entry_put_position"]["max_ask_above_bid_pct"])
            max_spread_str= f"{MAX_SPREAD_DECIMAL*100:>13.4f}%"                               

            # Strike Price Safety Margin Rule
            MIN_AVG_ABOVE_STRIKE_PCT = safe_percentage_to_float(rules["entry_put_position"]["min_avg_above_strike"])
            min_avg_above_strike_str = f"{MIN_AVG_ABOVE_STRIKE_PCT*100:>13.4f}%"

            REQUIRED_SMA_STRIKE_RATIO = 1.0 + MIN_AVG_ABOVE_STRIKE_PCT
            
            # Risk/Reward Ratio Rule
            MIN_RISK_REWARD_RATIO = float(rules["entry_put_position"]["min_risk_reward_ratio"])
            min_risk_reward_str = f"{MIN_RISK_REWARD_RATIO:>14.4f}"

            # Ranking flags (renamed from select_by to rank_by for clarity)
            RANK_BY_RR = bool(rules['entry_put_position'].get('rank_by_risk_reward_ratio', False))
            RANK_BY_ANNUAL = bool(rules['entry_put_position'].get('rank_by_annual_risk', False))
            RANK_BY_REV_ANN = bool(rules['entry_put_position'].get('rank_by_rev_ann_risk', False))
            RANK_BY_EXPECTED = bool(rules['entry_put_position'].get('rank_by_expected_profit', False))

            MIN_ANNUAL_RISK = float(rules['entry_put_position']['min_annual_risk_reward_ratio'])
            min_annual_risk_str = f"{MIN_ANNUAL_RISK:>14.4f}"
            # New: min_rev_annual_rr_ratio
            MIN_REV_ANNUAL_RISK = float(rules['entry_put_position'].get('min_rev_annual_rr_ratio', "-1000000"))
            min_rev_annual_risk_str = f"{MIN_REV_ANNUAL_RISK:>14.4f}"
            
            MIN_EXPECTED_PROFIT = safe_percentage_to_float(rules['entry_put_position']['min_expected_profit'])
            min_expected_profit_str= f"{MIN_EXPECTED_PROFIT*100:>13.4f}%"
            
            # Validate that only one ranking method is enabled at a time
            rankers_enabled = sum([1 if RANK_BY_RR else 0,
                                   1 if RANK_BY_ANNUAL else 0,
                                   1 if RANK_BY_REV_ANN else 0,
                                   1 if RANK_BY_EXPECTED else 0])
            if rankers_enabled > 1:
                print("❌ Rule error: More than one ranking method is enabled in rules.json. Please enable only one of rank_by_risk_reward_ratio, rank_by_annual_risk, rank_by_rev_ann_risk, rank_by_expected_profit.")
                return
            if rankers_enabled == 0:
                print("❌ Rule error: No ranking method is enabled in rules.json. Please enable one of rank_by_risk_reward_ratio, rank_by_annual_risk, rank_by_rev_ann_risk, rank_by_expected_profit.")
                return
            
            # Derive the per-trade premium budget from initial cash and account-level max positions
            # This replaces the hard-coded MAX_PREMIUM_PER_TRADE constant with a dynamic value.
            global MAX_PREMIUM_PER_TRADE
            try:
                if MAX_PUTS_PER_ACCOUNT > 0:
                    MAX_PREMIUM_PER_TRADE = float(INITIAL_CASH) / float(MAX_PUTS_PER_ACCOUNT)
                    max_premium_per_trade_str = f"{MAX_PREMIUM_PER_TRADE:>13.4f}"
                else:
                    # Fallback to existing constant if the rules are invalid
                    MAX_PREMIUM_PER_TRADE = MAX_PREMIUM_PER_TRADE
            except Exception:
                # In case of any unexpected parsing issues, keep the module-default
                pass

            # Define POSITION_STOP_LOSS_PCT first
            _pos_raw = rules.get('exit_put_position', {}).get('position_stop_loss_pct', "0%")
            POSITION_STOP_LOSS_PCT = abs(safe_percentage_to_float(_pos_raw)) if _pos_raw is not None else 0.0
            position_stop_loss_str = f"{POSITION_STOP_LOSS_PCT*100:>13.4f}%"

            # New Exit Rule: Min Gain to Take Profit (percentage profit threshold vs entry bid)
            try:
                _tp_raw = rules.get('exit_put_position', {}).get('min_gain_to_take_profit', None)
                TAKE_PROFIT_MIN_GAIN_PCT = safe_percentage_to_float(_tp_raw) if _tp_raw is not None else None
                take_profit_min_gain_str = f"{TAKE_PROFIT_MIN_GAIN_PCT*100:>13.4f}%"
            except Exception:
                TAKE_PROFIT_MIN_GAIN_PCT = None

            # New Exit Rule: Stock Min Above Strike (percentage buffer above strike required)
            try:
                _min_above_strike_raw = rules.get('exit_put_position', {}).get('stock_min_above_strike', None)
                STOCK_MIN_ABOVE_STRIKE_PCT = safe_percentage_to_float(_min_above_strike_raw) if _min_above_strike_raw is not None else None
                stock_min_above_strike_str = f"{STOCK_MIN_ABOVE_STRIKE_PCT*100:>13.4f}%"
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
                stock_max_below_entry_str = f"{STOCK_MAX_BELOW_ENTRY_PCT*100:>13.4f}%"
            except Exception:
                STOCK_MAX_BELOW_ENTRY_PCT = None

            # --- LOW PUT MODE RULES (optional block) ---
            try:
                low_rules = rules.get('low_put_mode', {})
                LOW_MIN_PUTS_TO_SET_LOW_MODE = int(low_rules.get('low_min_puts_to_set_low_mode')) if low_rules.get('low_min_puts_to_set_low_mode') is not None else None
                LOW_MAX_PUTS_TO_SET_HIGH_MODE = int(low_rules.get('low_max_puts_to_set_high_mode')) if low_rules.get('low_max_puts_to_set_high_mode') is not None else None
                LOW_MAX_PUTS_PER_ACCOUNT = int(low_rules.get('low_max_puts_per_account')) if low_rules.get('low_max_puts_per_account') is not None else None
                LOW_MAX_PUTS_PER_STOCK = int(low_rules.get('low_max_puts_per_stock')) if low_rules.get('low_max_puts_per_stock') is not None else None
                LOW_MAX_PUTS_PER_DAY = int(low_rules.get('low_max_puts_per_day')) if low_rules.get('low_max_puts_per_day') is not None else None

                LOW_MIN_5_DAY_RISE_PCT = safe_percentage_to_float(low_rules.get('low_min_5_day_rise_pct')) if low_rules.get('low_min_5_day_rise_pct') is not None else None
                LOW_MIN_ABOVE_AVG_PCT = safe_percentage_to_float(low_rules.get('low_min_above_avg_pct')) if low_rules.get('low_min_above_avg_pct') is not None else None
                LOW_MAX_ABOVE_AVG_PCT = safe_percentage_to_float(low_rules.get('low_max_above_avg_pct')) if low_rules.get('low_max_above_avg_pct') is not None else None
                LOW_MIN_AVG_UP_SLOPE_PCT = safe_percentage_to_float(low_rules.get('low_min_avg_up_slope_pct')) if low_rules.get('low_min_avg_up_slope_pct') is not None else None

                try:
                    LOW_MIN_STOCK_PRICE = float(str(low_rules.get('low_min_stock_price', '')).replace('$', '').replace(',', '').strip()) if low_rules.get('low_min_stock_price') is not None else None
                except Exception:
                    LOW_MIN_STOCK_PRICE = None

                LOW_MAX_DAYS_FOR_EXPIRATION = int(low_rules.get('low_max_days_for_expiration')) if low_rules.get('low_max_days_for_expiration') is not None else None

                LOW_MIN_PUT_BID_PRICE = float(low_rules.get('low_min_put_bid_price')) if low_rules.get('low_min_put_bid_price') is not None else None
                LOW_MIN_PUT_DELTA = safe_percentage_to_float(low_rules.get('low_min_put_delta')) if low_rules.get('low_min_put_delta') is not None else None
                LOW_MAX_PUT_DELTA = safe_percentage_to_float(low_rules.get('low_max_put_delta')) if low_rules.get('low_max_put_delta') is not None else None
                LOW_MAX_ASK_ABOVE_BID_PCT = safe_percentage_to_float(low_rules.get('low_max_ask_above_bid_pct')) if low_rules.get('low_max_ask_above_bid_pct') is not None else None

                LOW_MIN_RISK_REWARD_RATIO = float(low_rules.get('low_min_risk_reward_ratio')) if low_rules.get('low_min_risk_reward_ratio') is not None else None
                LOW_MIN_ANNUAL_RISK_REWARD_RATIO = float(low_rules.get('low_min_annual_risk_reward_ratio')) if low_rules.get('low_min_annual_risk_reward_ratio') is not None else None
                LOW_MIN_REV_ANNUAL_RR_RATIO = float(low_rules.get('low_min_rev_annual_rr_ratio')) if low_rules.get('low_min_rev_annual_rr_ratio') is not None else None
                LOW_MIN_EXPECTED_PROFIT = safe_percentage_to_float(low_rules.get('low_min_expected_profit')) if low_rules.get('low_min_expected_profit') is not None else None

                LOW_POSITION_STOP_LOSS_PCT = abs(safe_percentage_to_float(low_rules.get('low_position_stop_loss_pct'))) if low_rules.get('low_position_stop_loss_pct') is not None else None
                LOW_STOCK_MAX_BELOW_AVG = abs(safe_percentage_to_float(low_rules.get('low_stock_max_below_avg'))) if low_rules.get('low_stock_max_below_avg') is not None else None
                LOW_STOCK_MAX_BELOW_ENTRY = abs(safe_percentage_to_float(low_rules.get('low_stock_max_below_entry'))) if low_rules.get('low_stock_max_below_entry') is not None else None
                LOW_MIN_GAIN_TO_TAKE_PROFIT = safe_percentage_to_float(low_rules.get('low_min_gain_to_take_profit')) if low_rules.get('low_min_gain_to_take_profit') is not None else None

                # Formatted strings for reporting (consistent with other formatted strings)
                low_min_5_day_rise_str = f"{LOW_MIN_5_DAY_RISE_PCT*100:>13.4f}%" if LOW_MIN_5_DAY_RISE_PCT is not None else f"{'N/A':>14}"
                low_min_above_str = f"{LOW_MIN_ABOVE_AVG_PCT*100:>13.4f}%" if LOW_MIN_ABOVE_AVG_PCT is not None else f"{'N/A':>14}"
                low_max_above_str = f"{LOW_MAX_ABOVE_AVG_PCT*100:>13.4f}%" if LOW_MAX_ABOVE_AVG_PCT is not None else f"{'N/A':>14}"
                low_min_avg_slope_str = f"{LOW_MIN_AVG_UP_SLOPE_PCT*100:>13.4f}%" if LOW_MIN_AVG_UP_SLOPE_PCT is not None else f"{'N/A':>14}"
                low_min_stock_price_str = f"$ {LOW_MIN_STOCK_PRICE:>12.4f}" if LOW_MIN_STOCK_PRICE is not None else f"{'N/A':>14}"
                low_min_put_bid_price_str = f"$ {LOW_MIN_PUT_BID_PRICE:>12.4f}" if LOW_MIN_PUT_BID_PRICE is not None else f"{'N/A':>14}"
                low_min_put_delta_str = f"{LOW_MIN_PUT_DELTA*100:>13.4f}%" if LOW_MIN_PUT_DELTA is not None else f"{'N/A':>14}"
                low_max_put_delta_str = f"{LOW_MAX_PUT_DELTA*100:>13.4f}%" if LOW_MAX_PUT_DELTA is not None else f"{'N/A':>14}"
                low_max_ask_above_bid_str = f"{LOW_MAX_ASK_ABOVE_BID_PCT*100:>13.4f}%" if LOW_MAX_ASK_ABOVE_BID_PCT is not None else f"{'N/A':>14}"
                low_min_expected_profit_str = f"{LOW_MIN_EXPECTED_PROFIT*100:>13.4f}%" if LOW_MIN_EXPECTED_PROFIT is not None else f"{'N/A':>14}"
                low_position_stop_loss_str = f"{LOW_POSITION_STOP_LOSS_PCT*100:>13.4f}%" if LOW_POSITION_STOP_LOSS_PCT is not None else f"{'N/A':>14}"
                low_stock_max_below_avg_str = f"{LOW_STOCK_MAX_BELOW_AVG*100:>13.4f}%" if LOW_STOCK_MAX_BELOW_AVG is not None else f"{'N/A':>14}"
                low_stock_max_below_entry_str = f"{LOW_STOCK_MAX_BELOW_ENTRY*100:>13.4f}%" if LOW_STOCK_MAX_BELOW_ENTRY is not None else f"{'N/A':>14}"
                low_min_gain_to_takes_profit_str = f"{LOW_MIN_GAIN_TO_TAKE_PROFIT*100:>13.4f}%" if LOW_MIN_GAIN_TO_TAKE_PROFIT is not None else f"{'N/A':>14}"
            except Exception:
                LOW_MIN_PUTS_TO_SET_LOW_MODE = None
                LOW_MAX_PUTS_TO_SET_HIGH_MODE = None
                LOW_MAX_PUTS_PER_ACCOUNT = None
                LOW_MAX_PUTS_PER_STOCK = None
                LOW_MAX_PUTS_PER_DAY = None
                LOW_MIN_5_DAY_RISE_PCT = None
                LOW_MIN_ABOVE_AVG_PCT = None
                LOW_MAX_ABOVE_AVG_PCT = None
                LOW_MIN_AVG_UP_SLOPE_PCT = None
                LOW_MIN_STOCK_PRICE = None
                LOW_MAX_DAYS_FOR_EXPIRATION = None
                LOW_MIN_PUT_BID_PRICE = None
                LOW_MIN_PUT_DELTA = None
                LOW_MAX_PUT_DELTA = None
                LOW_MAX_ASK_ABOVE_BID_PCT = None
                LOW_MIN_RISK_REWARD_RATIO = None
                LOW_MIN_ANNUAL_RISK_REWARD_RATIO = None
                LOW_MIN_REV_ANNUAL_RR_RATIO = None
                LOW_MIN_EXPECTED_PROFIT = None
                LOW_POSITION_STOP_LOSS_PCT = None
                LOW_STOCK_MAX_BELOW_AVG = None
                LOW_STOCK_MAX_BELOW_ENTRY = None
                LOW_MIN_GAIN_TO_TAKE_PROFIT = None

            # Precompute Underlying Stock rules formatted strings (used in multiple tables)
            try:
                u_rules = rules.get('underlying_stock', {})
                u_min_rise =  safe_percentage_to_float(u_rules.get('min_5_day_rise_pct'))
                u_min_above = safe_percentage_to_float(u_rules.get('min_above_avg_pct'))
                u_max_above = safe_percentage_to_float(u_rules.get('max_above_avg_pct'))
                u_min_slope = safe_percentage_to_float(u_rules.get('min_avg_up_slope_pct'))
                try:
                    u_min_price = float(str(u_rules.get('min_stock_price', '')).replace('$', '').replace(',', '').strip())
                except Exception:
                    u_min_price = None

                # Consistent formatting widths across summaries
                initial_cash_str = f"${float(rules['account_simulation']['initial_cash']):>13,.2f}"
                min_rise_str     = f"{u_min_rise*100:>13.4f}%"  if u_min_rise  is not None else f"{'N/A':>14}"
                min_above_str    = f"{u_min_above*100:>13.4f}%" if u_min_above is not None else f"{'N/A':>14}"
                max_above_str    = f"{u_max_above*100:>13.4f}%" if u_max_above is not None else f"{'N/A':>14}"
                min_slope_str    = f"{u_min_slope*100:>13.4f}%" if u_min_slope is not None else f"{'N/A':>14}"
                min_price_str    = f"$ {u_min_price:>12.4f}"    if u_min_price is not None else f"{'N/A':>14}"
            except Exception:
                # Provide fallbacks if rules are missing or malformed
                min_rise_str  = f"{'N/A':>14}"
                min_above_str = f"{'N/A':>14}"
                max_above_str = f"{'N/A':>14}"
                min_slope_str = f"{'N/A':>14}"
                min_price_str = f"{'N/A':>14}"

            # Print rules in formatted tables
            print("\n=== TRADING RULES SUMMARY ===\n")
            
            print(f"📊 Account Simulation Rules")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Start Date                 | {start_date_str:<14} |")
            print(f"| End Date (Early Exit)      | {end_date_str:<14} |")
            print(f"| Initial Cash               | {initial_cash_str} |")
            print(f"| Max Puts/Account           | {MAX_PUTS_PER_ACCOUNT:>14} |")
            print(f"| Max Puts/Stock             | {MAX_PUTS_PER_STOCK:>14} |")
            print(f"| Max Puts/Day               | {MAX_PUTS_PER_DAY:>14} |")
            print(f"| Wrapper Sweep +/- Step %   | {wrapper_sweep_pct_str:>14} |")
            print(f"| Drawdown Goal %            | {drawdown_goal_pct_str:>14} |")
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
            print(f"| Min DTE                    | {MIN_DTE_RULE:>14} |")
            print(f"| Max DTE                    | {MAX_DTE_RULE:>14} |")
            print(f"| Min Put Bid Price          | {min_bid_price_str} |")
            print(f"| Min Put Delta              | {min_delta_str} |")
            print(f"| Max Put Delta              | {max_delta_str} |")
            print(f"| Max Bid-Ask Spread         | {max_spread_str} |")            
            print(f"| Min Avg Above Strike       | {min_avg_above_strike_str} |")
            print(f"| Min Risk/Reward Ratio      | {min_risk_reward_str} |")
            print(f"| Min Annual Risk            | {min_annual_risk_str} |")
            print(f"| Min Rev Annual Risk        | {min_rev_annual_risk_str} |")
            print(f"| Min Expected Profit        | {min_expected_profit_str} |")
            print(f"| Rank By Risk/Reward Ratio  | {('Yes' if RANK_BY_RR else 'No'):>14} |")
            print(f"| Rank By Annual Risk        | {('Yes' if RANK_BY_ANNUAL else 'No'):>14} |")
            print(f"| Rank By Rev Annual Risk    | {('Yes' if RANK_BY_REV_ANN else 'No'):>14} |")
            print(f"| Rank By Expected Profit    | {('Yes' if RANK_BY_EXPECTED else 'No'):>14} |")
            print(f"|----------------------------|----------------|")
            print()

            # 4. Exit Put Position Rules
            print(f"📉 Exit Put Position Rules")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Position Stop Loss         | {position_stop_loss_str} |")
            print(f"| Stock Below SMA150         | {stock_max_below_avg_str} |")
            print(f"| Stock Min Above Strike     | {stock_min_above_strike_str} |")   
            print(f"| Stock Max Below Entry      | {stock_max_below_entry_str} |") 
            print(f"| Min Gain to Take Profit    | {take_profit_min_gain_str} |")
            print(f"|----------------------------|----------------|")
            print()                        

            # 6. Trading Costs and Limits
            print("💰 Trading Parameters")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Commission/Contract        | ${commission_per_contract_str} |")
            print(f"| Max Premium/Trade          | ${max_premium_per_trade_str} |")
            print(f"|----------------------------|----------------|")
            print()

            # 7. Low Put Mode Rules
            print("🔽 Low Put Mode Rules")
            print(f"|----------------------------|----------------|")
            print(f"| Parameter                  | Value          |")
            print(f"|----------------------------|----------------|")
            print(f"| Min Puts to Set Low Mode   | {LOW_MIN_PUTS_TO_SET_LOW_MODE if LOW_MIN_PUTS_TO_SET_LOW_MODE is not None else 'N/A':>14} |")
            print(f"| Max Puts to Set High Mode  | {LOW_MAX_PUTS_TO_SET_HIGH_MODE if LOW_MAX_PUTS_TO_SET_HIGH_MODE is not None else 'N/A':>14} |")    
            print(f"| Low Max Puts/Account       | {LOW_MAX_PUTS_PER_ACCOUNT if LOW_MAX_PUTS_PER_ACCOUNT is not None else 'N/A':>14} |")
            print(f"| Low Max Puts/Stock         | {LOW_MAX_PUTS_PER_STOCK if LOW_MAX_PUTS_PER_STOCK is not None else 'N/A':>14} |")
            print(f"| Low Max Puts/Day           | {LOW_MAX_PUTS_PER_DAY if LOW_MAX_PUTS_PER_DAY is not None else 'N/A':>14} |")
            print(f"| Low Min 5-Day Rise         | {low_min_5_day_rise_str} |")
            print(f"| Low Min Above Avg          | {low_min_above_str} |")
            print(f"| Low Max Above Avg          | {low_max_above_str} |")
            print(f"| Low Min Avg Slope          | {low_min_avg_slope_str} |")
            print(f"| Low Min Stock Price        | {low_min_stock_price_str} |")
            print(f"| Low Max DTE                | {LOW_MAX_DAYS_FOR_EXPIRATION if LOW_MAX_DAYS_FOR_EXPIRATION is not None else 'N/A':>14} |")
            print(f"| Low Min Put Bid Price      | {low_min_put_bid_price_str} |")
            print(f"| Low Min Put Delta          | {low_min_put_delta_str} |")
            print(f"| Low Max Put Delta          | {low_max_put_delta_str} |")
            print(f"| Low Max Bid-Ask Spread     | {low_max_ask_above_bid_str} |")
            print(f"| Low Min Risk/Reward        | {LOW_MIN_RISK_REWARD_RATIO if LOW_MIN_RISK_REWARD_RATIO is not None else 'N/A':>14} |")
            print(f"| Low Min Annual Risk        | {LOW_MIN_ANNUAL_RISK_REWARD_RATIO if LOW_MIN_ANNUAL_RISK_REWARD_RATIO is not None else 'N/A':>14} |")
            print(f"| Low Min Rev Annual Risk    | {LOW_MIN_REV_ANNUAL_RR_RATIO if LOW_MIN_REV_ANNUAL_RR_RATIO is not None else 'N/A':>14} |")
            print(f"| Low Min Expected Profit    | {low_min_expected_profit_str} |")
            print(f"| Low Position Stop Loss     | {low_position_stop_loss_str} |")
            print(f"| Low Stock Below SMA150     | {low_stock_max_below_avg_str} |")
            print(f"| Low Stock Max Below Entry  | {low_stock_max_below_entry_str} |")
            print(f"| Low Min Gain Take Profit   | {low_min_gain_to_take_profit_str} |")
            print(f"|----------------------------|----------------|")
            print()

    # Compute runtime for the Performance Summary      
    _elapsed_seconds = int(time.perf_counter() - _sim_start_time)
    _hh = _elapsed_seconds // 3600
    _mm = (_elapsed_seconds % 3600) // 60
    _ss = _elapsed_seconds % 60
    runtime_str = f"{_hh:02d}:{_mm:02d}:{_ss:02d}"

    # Calculate Win Ratio (using incremental counters)
    win_ratio_pct = (winning_trades_count / closed_trades_count * 100.0) if closed_trades_count > 0 else 0.0
    
    # Performance Summary
    print("📊 Final Performance")
    print(f"|-----------------------------|-------------------------|") 
    print(f"| Parameter                   |  Value                  |")
    print(f"|-----------------------------|-------------------------|")
    print(f"| Current Date/Time           | {datetime.now().strftime('%Y-%m-%d %H:%M'):>23} |")
    print(f"| Annualized Gain             | {annualized_gain:>22.3f}% |")
    print(f"| Total Gain                  | ${TOTAL_GAIN:>22,.2f} |")    
    print(f"| Run Time                    | {runtime_str:>23} |")
    # Print the worst year and its percentage gain
    if worst_year is not None and worst_year_pct is not None:
        print(f"| Worst Year (Gain %)         | {str(worst_year) + ':  ' + format(worst_year_pct, '.2f') + '%':>23} |")
    print(f"| Peak Open Positions         | {peak_open_positions:>23} |")
    print(f"| Min DTE (Open Positions)    | {min_DTE_result:>23} |")
    print(f"| Max DTE (Open Positions)    | {max_DTE_result:>23} |")
    print(f"| Total Entry Events          | {total_entry_events:>23} |")
    print(f"| Win Ratio                   | {win_ratio_pct:>22.2f}% |")
    
    # Print current log file name (row ~2418 request)
    try:
        current_log_path = getattr(sys.stdout, 'logfile', None)
        if current_log_path and hasattr(current_log_path, 'name'):
            log_filename_only = os.path.basename(current_log_path.name)
            print(f"| Log File                    | {log_filename_only:>23} |")
        else:
            # Fallback if stdout has been restored or structure changed
            print(f"| Log File                    | {'N/A':>23} |")
    except Exception:
        print(f"| Log File                    | {'ERR':>23} |")
    
    # Worst drawdown across all simulated dates
    try:
        print(f"| Worst Drawdown              | {worst_drawdown_pct:>22.3f}% |")
    except Exception:
        # If for any reason the metric isn't available, skip gracefully
        pass

    Score1 = annualized_gain / (drawdown_goal_pct - worst_drawdown_pct) 
    Score2 = (annualized_gain + worst_year_pct * 15) / (drawdown_goal_pct - worst_drawdown_pct) 
    Score3 =  (annualized_gain * worst_year_pct ) / (drawdown_goal_pct - worst_drawdown_pct) 
    Score4 =  (annualized_gain * worst_year_pct * abs(worst_year_pct/20) ** 5 ) / (drawdown_goal_pct - worst_drawdown_pct) 
    Score5 =  (annualized_gain * abs(annualized_gain/60) ** 2  * worst_year_pct * abs(worst_year_pct/20) ** 5 ) / (drawdown_goal_pct - worst_drawdown_pct) 
    if annualized_gain < 0 and worst_year_pct < 0:
        Score5= -abs(Score5)

    if annualized_gain == 0 or worst_year_pct ==0:
        Score6 = -1e2*abs(worst_drawdown_pct)
    else:    
        Score6 =  annualized_gain * worst_year_pct / (drawdown_goal_pct - worst_drawdown_pct) 
        Score6 =  1e2 * Score6 * (abs(worst_year_pct)) ** 2 / abs(drawdown_goal_pct - worst_drawdown_pct) ** 2    
        if annualized_gain < 0 and worst_year_pct < 0:
            Score6= -abs(Score6)

    print(f"| Score Result                | {Score6:>23.4f} |")
    print(f"|-----------------------------|-------------------------|")
    print()    
# Execute the main function
if __name__ == "__main__":
    # Check if running from wrapper with custom rules path
    rules_path = os.environ.get("SIM_WRAPPER_RULES_PATH")
    if rules_path and os.path.exists(rules_path):
        load_and_run_simulation(rules_path, JSON_FILE_PATH)
    else:
        load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)