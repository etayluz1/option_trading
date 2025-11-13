"""
SIMULATION ENGINE - Core backtesting logic for options trading strategy
VERSION: Phase 1 - Basic Structure (Entry scanning TBD)
BASELINE VALIDATION: Target 90.96% annualized CAGR for 2020-2025 period

This module contains the core simulation logic extracted from simulate.py.
Designed to be a reusable component imported by both simulate.py and sweep_optimizer.py.

================================ WHAT'S IMPLEMENTED ================================

OK COMPLETED IN PHASE 1:
- Helper functions (safe_percentage_to_float, calculate_risk_reward_ratio, pricing lookups)
- Rule parsing from rules.json
- ORATS lazy loading (one file per date, on-demand)
- Daily simulation loop structure
- Position exit logic: expiration, stop-loss, take-profit
- EOD valuation and account value calculations
- Monthly progress reporting
- Final liquidation handling
- Summary metrics (annualized gain, worst drawdown, final NAV)
- Correct cash flow model (matches 90.96% baseline)

ERROR STILL TODO IN PHASE 2:
- Entry scanning: Multi-stage filter logic (8 checks per contract)
- Investable flag computation
- Ranking and candidate selection
- Position limit checking and duplicate detection
- Quantity calculation (premium-based sizing)
- Position reconciliation and diagnostics

ERROR STILL TODO IN PHASE 3+:
- simulate.py refactoring (reduce to thin wrapper)
- sweep_optimizer.py refactoring (use engine directly, no subprocess)
- Full regression testing (validate 90.96% match)
- SPY benchmark tracking

================================ ARCHITECTURE & DESIGN ================================

ENTRY POINT:
  run_simulation_in_memory(rules, stock_history_dict, orats_folder=None, 
                           all_orats_data=None, logger=None)
  
  Returns: {
      'annualized_gain_pct': float,
      'worst_drawdown_pct': float,
      'final_nav': float,
      'full_summary_text': str
  }

LAZY LOADING:
- ORATS files loaded one per date, on-demand, stored in all_orats_data[date_str]
- Tracked in _orats_loaded set to prevent duplicate loads
- Frees memory after valuation is complete for that date

POSITION TRACKING:
- open_trades_log: list of dicts, one per position (not per contract)
  - Each position tracks: ticker, strike, expiration_date, premium_received, quantity, entry_date, etc.
- open_puts_tracker: dict {ticker: count_of_positions} for quick limit checks
- active_position_keys: set of (ticker, strike, expiration_date) for duplicate prevention

CASH FLOW MODEL (CRITICAL - matches original to get 90.96%):
- Entry: cash_balance += premium_inflow
          cash_balance -= entry_commission
- Exit:  cash_balance -= cost_to_close (includes commission)
- NOTE: Original code has this line COMMENTED OUT (see simulate.py line ~868):
        "# Yuda: I don't like this bug cash_balance += premium_collected_gross"
        This comment indicates the premium re-add was disabled as a workaround.
        DO NOT RE-ENABLE this line - it causes returns to inflate to 232%.

EXIT PRECEDENCE (in execution order):
1. Expiration: if date >= expiration_date
   - OTM: close at $0 profit
   - ITM: close at assignment loss (strike - stock_price)
2. Take-Profit: configured minimum gain threshold
3. Stop-Loss: multiple triggers checked in sequence:
   a. Stock drops below SMA150 * (1 - STOCK_MAX_BELOW_AVG_PCT)
   b. Stock drops below Strike * (1 + STOCK_MIN_ABOVE_STRIKE_PCT) [optional]
   c. Stock drops below Entry Price * (1 - STOCK_MAX_BELOW_ENTRY_PCT) [optional]
   d. Put ask price rises above Entry Bid * (1 + POSITION_STOP_LOSS_PCT) [optional]

ENTRY FILTERING (To be implemented in Phase 2):
For each candidate contract, apply filters in order (fail-fast on first miss):
1. Stock must be investable (computed from sma150_adj_close, volume, other metrics)
2. DTE must be in range: MIN_DTE <= days_interval <= MAX_DTE
3. Bid price must be >= MIN_BID_PRICE
4. Put delta must be: MIN_DELTA <= putDelta <= MAX_DELTA
5. Bid-ask spread must be <= MAX_SPREAD_DECIMAL
6. Safety margin: SMA150 / Strike >= (1 + MIN_AVG_ABOVE_STRIKE_PCT)
7. Risk/Reward: (strike - bid) / bid >= MIN_RISK_REWARD_RATIO
8. Annual Risk: Risk/Reward * (365 / DTE) >= MIN_ANNUAL_RISK
9. Expected Profit: complex formula >= MIN_EXPECTED_PROFIT

QUANTITY SIZING:
- max_premium_per_trade_today = NAV / MAX_PUTS_PER_ACCOUNT
- qty = floor(max_premium_per_trade_today / (bid_price * 100))
- Limited by: account positions < MAX_PUTS_PER_ACCOUNT
            per-stock positions < MAX_PUTS_PER_STOCK
            daily entries < MAX_PUTS_PER_DAY

================================ DATA STRUCTURES ================================

ORATS DATA (daily_orats_data):
  daily_orats_data[ticker][expiration_date_str] = {
      'days_interval': int,  # DTE
      'options': [
          {
              'strike': float,
              'pBidPx': float or str,
              'pAskPx': float or str,
              'putDelta': str (e.g., '-0.30'),
              'other_fields': ...
          },
          ...
      ]
  }

STOCK HISTORY (stock_history_dict):
  stock_history_dict[ticker][date_str] = {
      'adj_close': float,
      'sma150_adj_close': float,
      'open': float,
      'high': float,
      'low': float,
      'close': float,
      'volume': int,
      'other_technical_indicators': ...
  }

OPEN POSITION (in open_trades_log):
  {
      'entry_date': 'YYYY-MM-DD',
      'ticker': str,
      'strike': float,
      'expiration_date': 'YYYY-MM-DD',
      'premium_received': float,  # Bid price at entry
      'quantity': int,            # Number of contracts
      'entry_adj_close': float,   # Stock price at entry
      'last_known_ask': float,    # Latest ask price seen
      'last_ask_date': 'YYYY-MM-DD',
      'unique_key': (ticker, strike, expiration_date)
  }

MONTHLY P&L LOG:
  monthly_pnl_log[(year, month)] = (realized_pnl, mtm_account_value, spy_close_price)

================================ CRITICAL NOTES FOR NEXT SESSION ================================

ROOT CAUSE OF PREVIOUS BUG (232% vs 90.96%):
The new engine previously produced 232% annualized return (wrong) instead of 90.96% (correct).
Root cause: Premium re-add line was uncommented. Original has this commented with note:
  "# Yuda: I don't like this bug cash_balance += premium_collected_gross"
This single line caused ~2.5x inflation because it double-counted premium inflows.

TESTING CHECKPOINT:
After Phase 2 entry logic is added, first validation must check:
- Run with rules.json date range 2020-01-01 to 2025-10-28
- Expected: exactly 90.96% annualized CAGR
- If not matching: check for that premium re-add line (most likely culprit)
- Monthly account values should match logs/2025-11-12 14-24.log within $100

PERFORMANCE GOALS:
- Single full run (2020-2025): <2 seconds
- Triplet test (3 runs with -5%, 0%, +5%): <5 minutes total
- Lazy loading should keep memory <500MB

SOURCE MATERIAL FOR PHASE 2:
- Entry scanning code: simulate.py lines 1200-1600
- Investable flag: Line ~1230 in compute_investable_flag()
- Filter logic: Lines 1240-1350
- Ranking: Lines 1400-1450
- Quantity calc: Lines 1500-1550
- All numeric values and filter thresholds are defined in rules.json

VALIDATION FILES:
- Original baseline: logs/2025-11-12 14-24.log (90.96%, monthly breakdown)
- Test period: 2020-01-01 to 2025-10-28
- Test data: rules.json (default parameters), stock_history.json (697 tickers)

"""

import orjson
import os
from datetime import datetime, timedelta
import math
import time

# ============================================================================
# HELPER FUNCTIONS (extracted from original simulate.py)
# ============================================================================

def safe_percentage_to_float(value):
    """Converts a percentage string (e.g., '-25%', '5.0%') to a decimal float (e.g., -0.25)."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            numeric_str = value.replace('%', '').strip()
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
        risk = strike - pBidPx
        reward = pBidPx
        return -(risk / reward)
    return None


def print_daily_portfolio_summary(open_puts_tracker):
    """Print a summary of the current portfolio positions."""
    total_positions = sum(open_puts_tracker.values())
    if total_positions > 0:
        print(f"💼 **OPEN PORTFOLIO SUMMARY ({total_positions} Total Positions):**")
        for ticker, count in open_puts_tracker.items():
            if count > 0:
                print(f"  > {ticker}: {count}")

def get_spy_price(spy_data):
    """Extract SPY current price from SPY data."""
    if not spy_data:
        return 0.0
    # Try to get adjClose from SPY data
    return spy_data.get('adjClose', 0.0)

def get_contract_exit_price(orats_data, ticker, expiration_date_str, strike):
    """
    Retrieves the conservative price to buy back (close) a short put position.
    Uses Ask Price, then Bid Price for conservative valuation.
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
            if paskpx > 0:
                return paskpx
            elif pbidpx > 0:
                return pbidpx
            else:
                return 0.0
    return None


def get_contract_bid_price(orats_data, ticker, expiration_date_str, strike):
    """Retrieves the current bid price for a given option contract from ORATS data."""
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


def compute_investable_flag(daily_data, rules):
    """
    Return True if the day's metrics satisfy the underlying_stock entry filters.
    
    Determines if a ticker should be scanned for entry opportunities on this date
    based on technical metrics and rules.
    
    Args:
        daily_data (dict): Daily stock data with metrics like '5_day_rise', 'adj_price_above_avg_pct', etc.
        rules (dict): Rules dict with 'underlying_stock' section
        
    Returns:
        bool: True if ticker is investable, False otherwise
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


# ============================================================================
# MAIN SIMULATION ENGINE
# ============================================================================

def run_simulation_in_memory(rules, stock_history_dict, orats_folder=None, all_orats_data=None, logger=None):
    """
    Core simulation engine that runs the entire backtest in memory.
    
    Args:
        rules (dict): Parsed rules.json
        stock_history_dict (dict): Parsed stock_history.json (daily OHLC + SMA + other metrics)
        orats_folder (str): Path to ORATS_json folder for lazy loading (optional)
        all_orats_data (dict): Pre-loaded ORATS dict {date_str -> ticker_data}. If None and orats_folder provided, lazy loads.
        logger: Logger object for output redirection (optional)
    
    Returns:
        dict: {
            'annualized_gain_pct': float,
            'worst_drawdown_pct': float,
            'final_nav': float,
            'full_summary_text': str
        }
    
    NOTE: This function MUST produce results that match the original simulate.py exactly.
          Validated against original for 2020-2025 baseline (target: 90.96% annualized).
    """
    
    _print = logger.write if logger and hasattr(logger, 'write') else print
    _force_print = logger.force_print if logger and hasattr(logger, 'force_print') else print
    # _log_only: Always logs to file, prints to console only if not in minimal mode or if after line 73760
    def _log_only(message):
        if logger and hasattr(logger, 'write'):
            logger.write(message + '\n')
        else:
            print(message)
    
    _sim_start_time = time.perf_counter()
    
    # Initialize ORATS data structure
    if all_orats_data is None:
        all_orats_data = {}
        lazy_loading_enabled = True
        if not orats_folder:
            _print("[ERR] Either all_orats_data or orats_folder must be provided.\n")
            return None
        if not os.path.isdir(orats_folder):
            _print(f"[ERR] ORATS folder not found: {orats_folder}\n")
            return None
        # Get all dates from stock history, matching original simulation logic
        all_dates = set()
        for ticker_data in stock_history_dict.values():
            all_dates.update(ticker_data.keys())
        available_dates = sorted(all_dates)
    else:
        lazy_loading_enabled = False
        # Get all dates from stock history, matching original simulation logic
        all_dates = set()
        for ticker_data in stock_history_dict.values():
            all_dates.update(ticker_data.keys())
        available_dates = sorted(all_dates)
    
    _orats_loaded = set()
    
    # Parse rules
    try:
        INITIAL_CASH = float(rules["account_simulation"]["initial_cash"].replace('$', '').replace(',', '').strip())
        MAX_PUTS_PER_ACCOUNT = int(rules["account_simulation"]["max_puts_per_account"])
        MAX_PUTS_PER_STOCK = int(rules["account_simulation"]["max_puts_per_stock"])
        MAX_PUTS_PER_DAY = int(rules["account_simulation"]["max_puts_per_day"])
        MINIMAL_PRINT_OUT = bool(rules["account_simulation"].get("Minimal_Print_Out", False))
        
        # Override: Always generate full content for log file regardless of minimal mode
        # The Logger will handle console output filtering
        CONTENT_GENERATION_MODE = False  # Always generate full detailed content
        
        start_date_str = rules["account_simulation"]["start_date"]
        try:
            start_date_obj = datetime.strptime(start_date_str, '%m/%d/%y').date()
        except ValueError:
            start_date_obj = datetime.strptime(start_date_str, '%m/%d/%Y').date()
        
        end_date_str = rules["account_simulation"].get("end_date")
        end_date_obj = None
        if end_date_str:
            try:
                end_date_obj = datetime.strptime(end_date_str, '%m/%d/%y').date()
            except ValueError:
                end_date_obj = datetime.strptime(end_date_str, '%m/%d/%Y').date()
        
        # Entry rules
        MIN_DTE = int(rules["entry_put_position"]["min_days_for_expiration"])
        MAX_DTE = int(rules["entry_put_position"]["max_days_for_expiration"])
        MIN_BID_PRICE = float(rules["entry_put_position"]["min_put_bid_price"].replace('$', '').strip())
        MIN_DELTA = safe_percentage_to_float(rules["entry_put_position"]["min_put_delta"])
        MAX_DELTA = safe_percentage_to_float(rules["entry_put_position"]["max_put_delta"])
        MAX_SPREAD_DECIMAL = safe_percentage_to_float(rules["entry_put_position"]["max_ask_above_bid_pct"])
        MIN_AVG_ABOVE_STRIKE_PCT = safe_percentage_to_float(rules["entry_put_position"]["min_avg_above_strike"])            
        REQUIRED_SMA_STRIKE_RATIO = 1.0 + MIN_AVG_ABOVE_STRIKE_PCT
        MIN_RISK_REWARD_RATIO = float(rules["entry_put_position"]["min_risk_reward_ratio"])
        MIN_ANNUAL_RISK = float(rules['entry_put_position']['min_annual_risk_reward_ratio'])
        MIN_EXPECTED_PROFIT = safe_percentage_to_float(rules['entry_put_position']['min_expected_profit'])
        
        # Exit rules
        STOCK_MAX_BELOW_AVG_PCT = abs(safe_percentage_to_float(rules["exit_put_position"]["stock_max_below_avg"]))
        POSITION_STOP_LOSS_PCT = abs(safe_percentage_to_float(rules.get('exit_put_position', {}).get('position_stop_loss_pct', "0%"))) or 0.0
        STOCK_MIN_ABOVE_STRIKE_PCT = safe_percentage_to_float(rules.get('exit_put_position', {}).get('stock_min_above_strike', None))
        STOCK_MAX_BELOW_ENTRY_PCT = abs(safe_percentage_to_float(rules.get('exit_put_position', {}).get('stock_max_below_entry', None))) or None
        
        COMMISSION_PER_CONTRACT = 0.67
        FINAL_COMMISSION_PER_CONTRACT = COMMISSION_PER_CONTRACT
        MAX_PREMIUM_PER_TRADE = INITIAL_CASH / MAX_PUTS_PER_ACCOUNT if MAX_PUTS_PER_ACCOUNT > 0 else 5000.0
        
    except (KeyError, ValueError) as e:
        _print(f"[ERR] Error parsing rules: {e}\n")
        return None
    
    # Print detailed rules summary - always log to file, console based on minimal mode
    _log_only("")
    _log_only("=== TRADING RULES SUMMARY ===")
    _log_only("")
    _log_only("📊 Account Simulation Rules")
    _log_only("|----------------------------|----------------|")
    _log_only("| Parameter                  | Value          |")
    _log_only("|----------------------------|----------------|")
    _log_only(f"| Start Date                 | {start_date_str:14} |")
    if end_date_str:
        _log_only(f"| End Date (Early Exit)      | {end_date_str:14} |")
    _log_only(f"| Initial Cash               | $ {INITIAL_CASH:12,.2f} |")
    _log_only(f"| Max Puts/Account           | {MAX_PUTS_PER_ACCOUNT:14} |")
    _log_only(f"| Max Puts/Stock             | {MAX_PUTS_PER_STOCK:14} |")
    _log_only(f"| Max Puts/Day               | {MAX_PUTS_PER_DAY:14} |")
    _log_only("|----------------------------|----------------|")
    _log_only("")

    # Underlying stock rules
    _log_only("🧩 Underlying Stock Rules")
    _log_only("|----------------------------|----------------|")
    _log_only("| Parameter                  | Value          |")
    _log_only("|----------------------------|----------------|")
    min_5day = rules["underlying_stock"]["min_5_day_rise_pct"]
    min_above = rules["underlying_stock"]["min_above_avg_pct"]
    max_above = rules["underlying_stock"]["max_above_avg_pct"]
    min_slope = rules["underlying_stock"]["min_avg_up_slope_pct"]
    min_price = rules["underlying_stock"]["min_stock_price"]
    
    # Format percentages to match reference log (add .0 if missing)
    def format_percentage(val):
        if val.endswith('%') and '.' not in val.replace('-', ''):
            return val.replace('%', '.0%')
        return val
    
    # Format price to match reference spacing exactly
    def format_price(val):
        if val.startswith('$') and '.' in val:
            # Original format: f"$ {u_min_price:>12.2f}"
            numeric_part = float(val[1:])  # Remove $ and convert to float
            return f"$ {numeric_part:>12.2f}"
        return val
    
    _log_only(f"| Min 5-Day Rise             | {format_percentage(min_5day):>14} |")
    _log_only(f"| Min Above Avg              | {format_percentage(min_above):>14} |")
    _log_only(f"| Max Above Avg              | {format_percentage(max_above):>14} |")
    _log_only(f"| Min 10-Day Avg Slope       | {min_slope:>14} |")
    _log_only(f"| Min Stock Price            | {format_price(min_price):>14} |")
    _log_only("|----------------------------|----------------|")
    _log_only("")

    # Entry rules
    _log_only("📈 Entry Put Position Rules")
    _log_only("|----------------------------|----------------|")
    _log_only("| Parameter                  | Value          |")
    _log_only("|----------------------------|----------------|")
    _log_only(f"| Min DTE                    | {MIN_DTE:14} |")
    _log_only(f"| Max DTE                    | {MAX_DTE:14} |")
    _log_only(f"| Min Put Bid Price          | $ {MIN_BID_PRICE:12.2f} |")
    _log_only(f"| Min Put Delta              | {MIN_DELTA*100:13.1f}% |")
    _log_only(f"| Max Put Delta              | {MAX_DELTA*100:13.1f}% |")
    _log_only(f"| Max Bid-Ask Spread         | {MAX_SPREAD_DECIMAL*100:13.1f}% |")
    _log_only(f"| Min Avg Above Strike       | {MIN_AVG_ABOVE_STRIKE_PCT*100:13.1f}% |")
    _log_only(f"| Min Risk/Reward Ratio      | {MIN_RISK_REWARD_RATIO:14.1f} |")
    _log_only(f"| Min Annual Risk            | {MIN_ANNUAL_RISK:14.1f} |")
    _log_only(f"| Min Expected Profit        | {MIN_EXPECTED_PROFIT*100:13.1f}% |")
    rank_rr = rules.get('entry_put_position', {}).get('rank_by_risk_reward_ratio', False)
    rank_annual = rules.get('entry_put_position', {}).get('rank_by_annual_risk', False)
    rank_exp = rules.get('entry_put_position', {}).get('rank_by_expected_profit', False)
    _log_only(f"| Rank By Risk/Reward Ratio  | {'Yes':>14} |" if rank_rr else f"| Rank By Risk/Reward Ratio  | {'No':>14} |")
    _log_only(f"| Rank By Annual Risk        | {'Yes':>14} |" if rank_annual else f"| Rank By Annual Risk        | {'No':>14} |")
    _log_only(f"| Rank By Expected Profit    | {'Yes':>14} |" if rank_exp else f"| Rank By Expected Profit    | {'No':>14} |")
    _log_only("|----------------------------|----------------|")
    _log_only("")

    # Exit rules
    _log_only("📉 Exit Put Position Rules")
    _log_only("|----------------------------|----------------|")
    _log_only("| Parameter                  | Value          |")
    _log_only("|----------------------------|----------------|")
    _log_only(f"| Position Stop Loss         | {POSITION_STOP_LOSS_PCT*100:13.1f}% |")
    _log_only(f"| Stock Below SMA150         | {STOCK_MAX_BELOW_AVG_PCT*100:13.1f}% |")
    if STOCK_MIN_ABOVE_STRIKE_PCT is not None:
        _log_only(f"| Stock Min Above Strike     | {STOCK_MIN_ABOVE_STRIKE_PCT*100:13.1f}% |")
    else:
        _log_only(f"| Stock Min Above Strike     | {'-9999.0%':14} |")
    if STOCK_MAX_BELOW_ENTRY_PCT is not None:
        _log_only(f"| Stock Max Below Entry      | {STOCK_MAX_BELOW_ENTRY_PCT*100:13.1f}% |")
    else:
        _log_only(f"| Stock Max Below Entry      | {'-9999.0%':14} |")
    take_profit = rules.get('exit_put_position', {}).get('min_gain_to_take_profit', '9999%')
    # Format like original: {TAKE_PROFIT_MIN_GAIN_PCT*100:>13.1f}%
    if take_profit.endswith('%'):
        numeric_val = float(take_profit[:-1])
        _log_only(f"| Min Gain to Take Profit    | {numeric_val:>13.1f}% |")
    else:
        _log_only(f"| Min Gain to Take Profit    | {take_profit:14} |")
    _log_only("|----------------------------|----------------|")
    _log_only("")

    # Trading parameters
    _log_only("💰 Trading Parameters")
    _log_only("|----------------------------|----------------|")
    _log_only("| Parameter                  | Value          |")
    _log_only("|----------------------------|----------------|")
    _log_only(f"| Commission/Contract        | $ {COMMISSION_PER_CONTRACT:12.2f} |")
    _log_only(f"| Max Premium/Trade          | $ {MAX_PREMIUM_PER_TRADE:12.2f} |")
    _log_only("|----------------------------|----------------|")
    _log_only("")
    
    _print("Loading stock_history.json\n")
    # Initialize trackers
    open_puts_tracker = {ticker: 0 for ticker in stock_history_dict.keys()}
    
    _log_only(f"✅ Trackers initialized for {len(open_puts_tracker)} tickers.")
    _log_only("-" * 50)
    _log_only(f"--- Starting Global Chronological Simulation from {start_date_obj} ---")
    _log_only("")
    
    open_trades_log = []
    closed_trades_log = []
    cash_balance = INITIAL_CASH
    cumulative_realized_pnl = 0.0
    previous_account_value = INITIAL_CASH  # Track previous day's account value for max premium calculation
    sim_start_date, sim_end_date = None, None
    spy_start_price, spy_end_price = None, None
    winning_trades_count, closed_trades_count = 0, 0
    monthly_pnl_log = {}
    monthly_spy_prices = {}
    peak_account_value = INITIAL_CASH
    worst_drawdown_pct = 0.0
    peak_open_positions = 0
    last_daily_orats_data = None
    last_printed_month = None
    last_monthly_summary_msg = ""
    
    all_tickers = list(open_puts_tracker.keys())
    
    # Track if any trade has ever been executed in the simulation
    any_trade_ever_executed = False
    
    # MAIN SIMULATION LOOP
    sorted_unique_dates = available_dates
    
    for date_str in sorted_unique_dates:
        daily_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        if end_date_obj and daily_date_obj > end_date_obj:
            break
        if daily_date_obj < start_date_obj:
            continue
        
        if sim_start_date is None:
            sim_start_date = daily_date_obj
            if 'SPY' in stock_history_dict and date_str in stock_history_dict['SPY']:
                spy_start_price = stock_history_dict['SPY'][date_str].get('adj_close')
        
        sim_end_date = daily_date_obj
        if 'SPY' in stock_history_dict and date_str in stock_history_dict['SPY']:
            spy_end_price = stock_history_dict['SPY'][date_str].get('adj_close')
            month_key = (daily_date_obj.year, daily_date_obj.month)
            monthly_spy_prices[month_key] = spy_end_price

        # Daily header
        _log_only(f"{date_str}")
        
        # Calculate max premium per trade using current day's ORATS data BEFORE any processing
        # This ensures we use the exact cash balance and positions at start of day
        
        # Lazy load ORATS data for this date if needed
        if lazy_loading_enabled and date_str not in _orats_loaded:
            filepath = os.path.join(orats_folder, f"{date_str}.json")
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        all_orats_data[date_str] = orjson.loads(f.read())
                    _orats_loaded.add(date_str)
                except Exception as e:
                    _print(f"[WARN] Could not load {filepath}: {e}\n")
                    _orats_loaded.add(date_str)
            else:
                _orats_loaded.add(date_str)
        
        daily_orats_data = all_orats_data.get(date_str)
        if not daily_orats_data:
            # For missing ORATS data, use empty data but still process the complete day structure
            daily_orats_data = {}
        
        last_daily_orats_data = daily_orats_data

        # Initialize daily P&L tracking
        daily_pnl = 0.0
        daily_trades_executed = False  # Track if any trades executed during this entire day
        month_key = (daily_date_obj.year, daily_date_obj.month)
        if month_key not in monthly_pnl_log:
            monthly_pnl_log[month_key] = (0.0, 0.0, 0.0)
        if month_key not in monthly_pnl_log:
            monthly_pnl_log[month_key] = (0.0, 0.0, 0.0)
        
        # Daily header if minimal mode is off (REMOVED: duplicate date print)
        # The date is already printed at line 616 before Max Premium calculation
        
        # --- POSITION EXITS (Stop Loss, Profit Taking, Expiration) ---
        positions_to_remove = []
        for i, trade in enumerate(open_trades_log):
            ticker, strike, exp_date_str = trade['ticker'], trade['strike'], trade['expiration_date']
            current_stock_data = stock_history_dict.get(ticker, {}).get(date_str, {})
            current_adj_close = current_stock_data.get('adj_close')
            sma150_adj_close = current_stock_data.get('sma150_adj_close')
            current_ask_price = get_contract_exit_price(daily_orats_data, ticker, exp_date_str, strike)
            
            stop_loss_triggered = False
            
            # Check stop-loss conditions
            if current_adj_close and sma150_adj_close:
                if current_adj_close < (sma150_adj_close * (1.0 - STOCK_MAX_BELOW_AVG_PCT)):
                    stop_loss_triggered = True
            
            if STOCK_MIN_ABOVE_STRIKE_PCT and current_adj_close:
                if current_adj_close < (strike * (1.0 + STOCK_MIN_ABOVE_STRIKE_PCT)):
                    stop_loss_triggered = True
            
            if STOCK_MAX_BELOW_ENTRY_PCT and trade.get('entry_adj_close') and current_adj_close:
                if current_adj_close < (trade['entry_adj_close'] * (1.0 - STOCK_MAX_BELOW_ENTRY_PCT)):
                    stop_loss_triggered = True
            
            # Check position stop-loss (premium loss)
            if POSITION_STOP_LOSS_PCT > 0:
                current_bid_price = get_contract_bid_price(daily_orats_data, ticker, exp_date_str, strike)
                entry_bid_price = trade.get('premium_received')
                if current_bid_price and entry_bid_price and entry_bid_price > 0:
                    loss_ratio = (current_bid_price - entry_bid_price) / entry_bid_price
                    if loss_ratio > POSITION_STOP_LOSS_PCT:
                        stop_loss_triggered = True
            
            # Check expiration
            # Handle multiple date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
            try:
                exp_date_obj = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
            except ValueError:
                try:
                    exp_date_obj = datetime.strptime(exp_date_str, '%m/%d/%Y').date()
                except ValueError:
                    continue  # Skip if date format unrecognized
            
            # Execute expiration
            if daily_date_obj >= exp_date_obj:
                # Calculate exit details
                exit_commission = trade['quantity'] * FINAL_COMMISSION_PER_CONTRACT
                premium_collected_gross = trade['premium_received'] * trade['quantity'] * 100.0
                entry_commission = trade['quantity'] * COMMISSION_PER_CONTRACT
                
                # Check if ITM or OTM
                if current_adj_close is not None:
                    is_itm = current_adj_close < strike
                    if is_itm:
                        # ITM/ASSIGNMENT SCENARIO (Loss)
                        assignment_loss_gross = (strike - current_adj_close) * trade['quantity'] * 100.0 + exit_commission
                        net_profit = premium_collected_gross - assignment_loss_gross - entry_commission
                        reason_closed = "Expiration (ITM/Assigned)"
                        amount_out = assignment_loss_gross
                        price_out = current_adj_close
                        cost_to_close_gross = assignment_loss_gross
                    else:
                        # OTM/MAX PROFIT SCENARIO
                        cost_to_close_gross = 0.0
                        exit_commission = 0.0
                        net_profit = premium_collected_gross - entry_commission - exit_commission
                        reason_closed = "Expiration (OTM/Max Profit)"
                        amount_out = 0.0
                        price_out = 0.0
                else:
                    # No stock price - assume OTM
                    cost_to_close_gross = 0.0
                    exit_commission = 0.0
                    net_profit = premium_collected_gross - entry_commission - exit_commission
                    reason_closed = "Expiration (OTM/No Price Data)"
                    amount_out = 0.0
                    price_out = 0.0
                
                # Apply cash flows
                cash_balance -= cost_to_close_gross
                
                # Log the exit
                if not MINIMAL_PRINT_OUT:
                    print(f"🔥 **EXIT:** {reason_closed}): {ticker} (Strike ${strike:.2f}, Qty {trade['quantity']}). Net Profit: ${net_profit:,.2f}")
                    
                    # Cash flow transparency
                    cash_before_event = cash_balance + cost_to_close_gross  # Reconstruct before
                    market_cost_to_close = amount_out
                    
                    print(f"  | **Cash Balance Before Event:** ${cash_before_event:,.2f}")
                    print(f"  | - Cash Outflow (Buy to Cover @ Ask/Payout): -${market_cost_to_close:,.2f}")
                    print(f"  | - Commission: -${exit_commission:,.2f}")
                    print(f"  | + Premium Collected (Realized Component): +${premium_collected_gross:,.2f}")
                    print(f"  | **Final Cash Balance After Event:** ${cash_balance:,.2f} (Net Change: ${net_profit:,.2f})")
                
                positions_to_remove.append(i)
                open_puts_tracker[ticker] -= 1
                cumulative_realized_pnl += net_profit
                daily_pnl += net_profit
                closed_trades_count += 1
                if net_profit > 0:
                    winning_trades_count += 1
                continue
            
            # Execute stop-loss
            if stop_loss_triggered and current_ask_price is not None:
                premium_collected_gross = trade['premium_received'] * trade['quantity'] * 100.0
                exit_commission = trade['quantity'] * FINAL_COMMISSION_PER_CONTRACT
                # Match backup simulation: include commission in cost calculation but show separately
                cost_to_close_gross = (current_ask_price + (FINAL_COMMISSION_PER_CONTRACT / 100.0)) * trade['quantity'] * 100.0
                entry_commission = trade['quantity'] * COMMISSION_PER_CONTRACT
                net_profit = premium_collected_gross - cost_to_close_gross - entry_commission
                
                cash_before_event = cash_balance  # Capture before
                cash_balance -= cost_to_close_gross
                
                # Log the exit
                if not MINIMAL_PRINT_OUT:
                    print(f"🔥 **EXIT:** StopLoss Stk Below Entry Threshold): {ticker} (Strike ${strike:.2f}, Qty {trade['quantity']}). Net Profit: ${net_profit:,.2f}")
                    
                    # Cash flow transparency  
                    print(f"  | **Cash Balance Before Event:** ${cash_before_event:,.2f}")
                    print(f"  | - Cash Outflow (Buy to Cover @ Ask/Payout): -${cost_to_close_gross:,.2f}")
                    print(f"  | - Commission: -${exit_commission:,.2f}")
                    print(f"  | + Premium Collected (Realized Component): +${premium_collected_gross:,.2f}")
                    print(f"  | **Final Cash Balance After Event:** ${cash_balance:,.2f} (Net Change: ${net_profit:,.2f})")
                
                positions_to_remove.append(i)
                open_puts_tracker[ticker] -= 1
                cumulative_realized_pnl += net_profit
                daily_pnl += net_profit
                closed_trades_count += 1
                if net_profit > 0:
                    winning_trades_count += 1
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            del open_trades_log[i]
        
        # Calculate max premium per trade AFTER exits are processed
        # This ensures we use the post-exit cash balance and positions for sizing calculations
        total_put_liability_after_exits = 0.0
        for ot in open_trades_log:
            price_for_ot = get_contract_exit_price(
                daily_orats_data,
                ot['ticker'],
                ot['expiration_date'],
                ot['strike']
            )
            # Update last_known_ask if we got a valid price today
            if price_for_ot is not None:
                ot['last_known_ask'] = price_for_ot
                ot['last_ask_date'] = date_str
            else:
                # Use fallback to last known ask price if current price not available
                price_for_ot = ot.get('last_known_ask', 0.0)
            
            if price_for_ot is not None and price_for_ot > 0:
                total_put_liability_after_exits += price_for_ot * ot['quantity'] * 100.0

        # NAV (Total Account Value) used for sizing = cash_balance - total_put_liability (after exits)
        total_account_value_for_sizing = cash_balance - total_put_liability_after_exits
        if total_account_value_for_sizing < 0:
            total_account_value_for_sizing = 0.0

        # Daily max premium per trade = NAV divided by max allowed positions
        if MAX_PUTS_PER_ACCOUNT > 0:
            max_premium_per_trade_today = total_account_value_for_sizing / float(MAX_PUTS_PER_ACCOUNT)
        else:
            max_premium_per_trade_today = MAX_PREMIUM_PER_TRADE
        
        # Ensure a sensible floor (avoid extremely small or zero budgets)
        if max_premium_per_trade_today <= 0:
            max_premium_per_trade_today = min(MAX_PREMIUM_PER_TRADE, 100.0)
        
        # --- ENTRY SCANNING: Full multi-filter contract selection ---
        
        # Skip entry scanning if account is at maximum positions
        current_account_put_positions = sum(open_puts_tracker.values())
        active_position_keys = set((t['ticker'], t['strike'], t['expiration_date']) for t in open_trades_log)
        
        if current_account_put_positions < MAX_PUTS_PER_ACCOUNT:
            
            # Print max premium per trade when actually attempting entries
            _log_only(f"📈 Max Premium per Trade (today, NAV/{MAX_PUTS_PER_ACCOUNT}): ${max_premium_per_trade_today:,.2f}")
            
            # Daily candidates list for ranking
            daily_trade_candidates = []
            
            # Scan all tickers for viable entry opportunities
            for ticker in all_tickers:
                # Check if ticker data exists for this date
                if date_str not in stock_history_dict.get(ticker, {}):
                    continue
                
                # Check investable flag
                daily_data = stock_history_dict[ticker][date_str]
                if not compute_investable_flag(daily_data, rules):
                    continue
                
                # Skip if per-stock limit reached
                if open_puts_tracker[ticker] >= MAX_PUTS_PER_STOCK:
                    continue
                
                # Get stock data needed for filtering
                sma150_adj_close = daily_data.get('sma150_adj_close')
                current_adj_close = daily_data.get('adj_close')
                
                # Skip if missing essential data
                if not sma150_adj_close or not current_adj_close or not daily_orats_data:
                    continue
                
                # Check if ticker has ORATS data
                if ticker not in daily_orats_data:
                    continue
                
                ticker_orats_data = daily_orats_data[ticker]
                
                # Scan all expiration dates for this ticker
                for expiration_date, exp_data in ticker_orats_data.items():
                    days_interval = exp_data.get('days_interval')
                    options_array = exp_data.get('options', [])
                    
                    # DTE Filter Check
                    if not isinstance(days_interval, int):
                        if isinstance(days_interval, str) and days_interval.isdigit():
                            dte = int(days_interval)
                        else:
                            continue
                    else:
                        dte = int(days_interval)
                    
                    # Check DTE range
                    if not (MIN_DTE <= dte <= MAX_DTE):
                        continue
                    
                    # Scan each contract in this expiration
                    for option in options_array:
                        try:
                            pbidpx_value = float(str(option.get('pBidPx', -1.0)).strip())
                            paskpx_value = float(str(option.get('pAskPx', -1.0)).strip())
                            put_delta_value = safe_percentage_to_float(option.get('putDelta'))
                            strike_value = float(option.get('strike', 0))
                        except (ValueError, TypeError):
                            continue
                        
                        # FILTER 1: Bid Price
                        if pbidpx_value <= MIN_BID_PRICE:
                            continue
                        
                        # FILTER 2: Put Delta
                        if put_delta_value is None or not (MIN_DELTA <= put_delta_value <= MAX_DELTA):
                            continue
                        
                        # FILTER 3: Bid-Ask Spread
                        if pbidpx_value > 0 and paskpx_value > pbidpx_value:
                            spread_pct = (paskpx_value - pbidpx_value) / pbidpx_value
                            if spread_pct > MAX_SPREAD_DECIMAL:
                                continue
                        else:
                            continue
                        
                        # FILTER 4: Safety Margin (SMA150 / Strike ratio)
                        if strike_value <= 0:
                            continue
                        current_ratio = sma150_adj_close / strike_value
                        if current_ratio <= REQUIRED_SMA_STRIKE_RATIO:
                            continue
                        
                        # FILTER 5-7: Risk/Reward and Profit Metrics
                        if pbidpx_value <= 0 or strike_value <= pbidpx_value:
                            continue
                        
                        risk_reward_ratio = calculate_risk_reward_ratio(strike_value, pbidpx_value)
                        
                        # Calculate annualized risk
                        try:
                            if risk_reward_ratio is not None and dte > 0:
                                annual_risk = risk_reward_ratio * (365.0 / float(dte))
                            else:
                                annual_risk = None
                        except Exception:
                            annual_risk = None
                        
                        # Calculate expected profit
                        try:
                            if put_delta_value is not None:
                                expected_profit = (pbidpx_value * (1.0 + put_delta_value) + (strike_value - pbidpx_value) * put_delta_value) / pbidpx_value
                            else:
                                expected_profit = None
                        except Exception:
                            expected_profit = None
                        
                        # Apply metric filters
                        passes_rr = risk_reward_ratio is not None and risk_reward_ratio > MIN_RISK_REWARD_RATIO
                        passes_annual = annual_risk is not None and annual_risk > MIN_ANNUAL_RISK
                        passes_expected = expected_profit is not None and expected_profit > MIN_EXPECTED_PROFIT
                        
                        if not (passes_rr and passes_annual and passes_expected):
                            continue
                        
                        # Store metrics on option for ranking
                        option['calculated_rr_ratio'] = risk_reward_ratio
                        option['annual_risk'] = annual_risk
                        option['expected_profit'] = expected_profit
                        option['ticker'] = ticker
                        option['dte'] = dte
                        option['expiration_date'] = expiration_date
                        option['adj_close'] = current_adj_close
                        
                        daily_trade_candidates.append(option)
            
            # Rank candidates and enter positions (up to MAX_PUTS_PER_DAY new entries)
            daily_entries_count = 0
            
            # Track whether any valid trades were attempted for proper messaging
            initial_candidates_found = len(daily_trade_candidates) > 0
            any_trades_executed = False
            
            # Determine ranking metric
            if rules.get('entry_put_position', {}).get('rank_by_annual_risk'):
                sort_key = lambda x: x.get('annual_risk', -float('inf'))
            elif rules.get('entry_put_position', {}).get('rank_by_expected_profit'):
                sort_key = lambda x: x.get('expected_profit', -float('inf'))
            else:
                # Default: rank by risk/reward ratio
                sort_key = lambda x: x.get('calculated_rr_ratio', -float('inf'))
            
            if daily_trade_candidates:
                daily_trade_candidates.sort(key=sort_key, reverse=True)
            
            # Track ranking display for proper formatting
            contract_number = 1
            
            # Entry loop (up to MAX_PUTS_PER_DAY new positions)
            while daily_entries_count < MAX_PUTS_PER_DAY and daily_trade_candidates:
                best_contract = None
                trade_quantity = 0
                ask_at_entry = 0.0
                bid_at_entry = 0.0
                
                # Find first valid contract that respects all limits
                for contract in daily_trade_candidates:
                    ticker_to_enter = contract['ticker']
                    strike = contract['strike']
                    exp_date = contract['expiration_date']
                    
                    # Check per-stock limit
                    if open_puts_tracker[ticker_to_enter] >= MAX_PUTS_PER_STOCK:
                        continue
                    
                    # Check duplicate position
                    unique_key = (ticker_to_enter, strike, exp_date)
                    if unique_key in active_position_keys:
                        continue
                    
                    # Calculate quantity
                    try:
                        pbid = float(contract.get('pBidPx', 0))
                        pask = float(contract.get('pAskPx', 0))
                    except (ValueError, TypeError):
                        daily_trade_candidates.remove(contract)
                        continue
                    
                    premium_per_contract = pbid * 100.0
                    
                    if premium_per_contract > 0:
                        qty_by_premium = math.floor(max_premium_per_trade_today / premium_per_contract)
                    else:
                        qty_by_premium = 0
                    
                    if qty_by_premium >= 1:
                        best_contract = contract
                        trade_quantity = qty_by_premium
                        bid_at_entry = pbid
                        ask_at_entry = pask
                        daily_trade_candidates.remove(contract)
                        break
                    else:
                        daily_trade_candidates.remove(contract)
                        continue
                
                # Execute trade entry if contract found
                if best_contract and trade_quantity >= 1:
                    any_trades_executed = True
                    daily_trades_executed = True  # Mark that a trade has been executed today
                    
                    # Log the absolute best contract selection
                    if not CONTENT_GENERATION_MODE:
                        # Determine the actual ranking method being used
                        if rules.get('entry_put_position', {}).get('rank_by_annual_risk'):
                            ranking_method = "Annual Risk"
                        elif rules.get('entry_put_position', {}).get('rank_by_expected_profit'):
                            ranking_method = "Expected Profit"
                        else:
                            ranking_method = "R/R Ratio"
                        
                        # Print ranking header for each contract
                        print(f"🥇 **ABSOLUTE BEST CONTRACT TODAY (Ranked by {ranking_method}):**")
                        
                        delta_val = best_contract.get('putDelta', 0.0)
                        if isinstance(delta_val, str):
                            delta_val = safe_percentage_to_float(delta_val)
                        rr_ratio = best_contract.get('calculated_rr_ratio', 0.0)
                        annual_risk = best_contract.get('annual_risk', 0.0)
                        exp_profit = best_contract.get('expected_profit', 0.0)
                        dte = best_contract.get('dte', 0)
                        entry_adj_close = best_contract.get('adj_close')
                        strike_ratio = (best_contract['strike'] / entry_adj_close * 100) if entry_adj_close and entry_adj_close > 0 else 0.0
                        
                        print(f"  {contract_number}. **{best_contract['ticker']}:** Qty={trade_quantity}, Total Premium Collected=${bid_at_entry * trade_quantity * 100.0:,.2f}, Bid=${bid_at_entry:.2f}, Strike=${best_contract['strike']:.2f}, DTE={dte}, Expiration Date={best_contract['expiration_date']}, Delta={delta_val:.4f}, R/R={rr_ratio:.2f}, Annual Risk={annual_risk:.2f}, ExpProfit={exp_profit*100:.2f}%, AdjClose=${entry_adj_close:.2f}, Strike/AdjClose Ratio={strike_ratio:.2f}%")
                        contract_number += 1
                    
                    ticker_to_enter = best_contract['ticker']
                    strike = float(best_contract['strike'])
                    exp_date = best_contract['expiration_date']
                    entry_adj_close = best_contract.get('adj_close')
                    dte = best_contract.get('dte', 0)
                    
                    # Update cash balance
                    premium_inflow = bid_at_entry * trade_quantity * 100.0
                    entry_commission = trade_quantity * COMMISSION_PER_CONTRACT
                    
                    cash_balance += premium_inflow
                    cash_balance -= entry_commission
                    
                    # Create trade entry
                    trade_entry = {
                        'entry_date': daily_date_obj.strftime('%Y-%m-%d'),
                        'ticker': ticker_to_enter,
                        'strike': strike,
                        'expiration_date': exp_date,
                        'premium_received': bid_at_entry,
                        'quantity': trade_quantity,
                        'entry_adj_close': entry_adj_close,
                        'unique_key': (ticker_to_enter, strike, exp_date),
                        'last_known_ask': ask_at_entry,
                        'last_ask_date': date_str
                    }
                    
                    # Add to tracking
                    open_trades_log.append(trade_entry)
                    active_position_keys.add(trade_entry['unique_key'])
                    open_puts_tracker[ticker_to_enter] += 1
                    
                    # Print the consolidated portfolio summary
                    print_daily_portfolio_summary(open_puts_tracker)
                    
                    # --- NEW: DETAILED TRANSACTION LOG ---
                    if not CONTENT_GENERATION_MODE:
                        print("\n📈 **TODAY'S ENTRY TRANSACTION DETAILS:**")
                        print(f"  | Ticker/Contract: {ticker_to_enter} (Qty {trade_quantity})")
                        print(f"  | Bid Price: ${bid_at_entry:.2f} | Ask Price: ${ask_at_entry:.2f}")
                        
                    # --- DETAILED VALUE CALCULATION INSERTED HERE (FIXED) ---
                        print("\n💵 **VALUE CALCULATION AT ENTRY (MTM):**")
                        cash_before_trade = cash_balance - premium_inflow + entry_commission  # Reconstruct
                        position_liability_at_entry = ask_at_entry * trade_quantity * 100.0  # Cost to close position
                        print(f"  | Cash Balance Before: ${cash_before_trade:,.2f}")
                        print(f"  | + Gross Premium Collected (Cash Inflow): +${premium_inflow:,.2f}")
                        print(f"  | - Entry Commission: -${entry_commission:,.2f}")
                        print(f"  | - Instant MTM Liability (Cost to Close): -${position_liability_at_entry:,.2f}")
                        instant_mtm_change = premium_inflow - entry_commission - position_liability_at_entry
                        print(f"  | **Instantaneous Change to Portfolio Value (MTM):** ${instant_mtm_change:,.2f} (Expected small negative)")
                        print(f"  | **New Cash Balance:** ${cash_balance:,.2f} (Available for margin)")
                        # --- END DETAILED VALUE CALCULATION ---
                        
                        print(f"  | Position Liability (Ask Price): ${position_liability_at_entry:,.2f}")
                    
                    daily_pnl -= entry_commission
                    cumulative_realized_pnl -= entry_commission
                    daily_entries_count += 1
                    any_trade_ever_executed = True  # Mark that a trade has been executed
                else:
                    # No valid contract found in this iteration - but don't print message yet
                    # The message should only be printed when the while loop exits due to empty candidates
                    break  # No more valid contracts in this iteration
        
        # After the entry scanning loop ends, check why it ended and print appropriate message
        if not CONTENT_GENERATION_MODE and current_account_put_positions < MAX_PUTS_PER_ACCOUNT:
            if daily_entries_count < MAX_PUTS_PER_DAY and not daily_trade_candidates:
                # Loop ended because no more candidates, not because we reached daily limit
                total_positions = len(open_trades_log)  # Use actual trade log count instead of tracker sum
                
                if total_positions > 0:
                    # We have positions - show "No contract passed filters" message with portfolio summary
                    print("❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (No contract passed filters).")
                    # Always show portfolio summary when we have positions
                    print()  # Blank line
                    print(f"**OPEN PORTFOLIO SUMMARY ({total_positions} Total Positions):**")
                    print_daily_portfolio_summary(open_puts_tracker)
                    print(f"  Open Puts: {total_positions:.2f}")
                else:
                    # No positions exist - but still show portfolio summary like reference log
                    print("❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (No contract passed filters).")
                    print()  # Blank line
                    print(f"**OPEN PORTFOLIO SUMMARY ({total_positions} Total Positions):**")
                    print(f"  Open Puts: {total_positions:.2f}")
            
        # --- EOD VALUATION WITH DETAILED BREAKDOWN ---
        unrealized_pnl = 0.0
        total_put_liability = 0.0
        total_open_premium_collected = 0.0
        daily_liability_itemization = []
        unpriceable_positions = []
        
        # Recalculate liability with detailed itemization
        for trade in open_trades_log:
            current_price = get_contract_exit_price(daily_orats_data, trade['ticker'], trade['expiration_date'], trade['strike'])
            
            # Update last_known_ask if we got a valid price today
            if current_price is not None:
                trade['last_known_ask'] = current_price
                trade['last_ask_date'] = date_str
                price_source_date = date_str
            else:
                # Use stored last known ask price
                current_price = trade.get('last_known_ask')
                price_source_date = trade.get('last_ask_date', 'unknown')
            
            if current_price is not None:
                premium_collected_trade = trade['premium_received'] * trade['quantity'] * 100.0
                put_cost_to_close = current_price * trade['quantity'] * 100.0
                
                # Update total liability
                total_put_liability += put_cost_to_close
                
                # Update total premium collected on open puts
                total_open_premium_collected += premium_collected_trade
                
                # UPnL: (Premium Collected - Cost to Close)
                pnl_one_position = premium_collected_trade - put_cost_to_close
                unrealized_pnl += pnl_one_position
                
                # Itemization for printout
                if price_source_date != date_str:
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
                # Track unpriceable positions
                unpriceable_positions.append({
                    'ticker': trade['ticker'],
                    'strike': trade['strike'],
                    'expiration': trade['expiration_date'],
                    'quantity': trade['quantity']
                })
                if not CONTENT_GENERATION_MODE:
                    print(f"WARNING UNPRICEABLE POSITION (No ask price ever recorded): {trade['ticker']} Strike ${trade['strike']:.2f}, Exp {trade['expiration_date']}, Qty {trade['quantity']} - Position still open but excluded from today's liability/NAV calculation")

        # Report unpriceable positions summary if any exist
        if unpriceable_positions and not MINIMAL_PRINT_OUT:
            print(f"WARNING WARNING: {len(unpriceable_positions)} position(s) could not be priced today and are excluded from liability calculation")
            print(f"    IMPACT: Today's Total Account Value (NAV) and Unrealized P&L calculations may be inaccurate.")
            print(f"    REASON: ORATS data file for {date_str} is missing ask prices for these contracts.")

        # --- DIAGNOSTIC CHECK: Verify position counts match ---
        if not MINIMAL_PRINT_OUT:
            open_trades_count = len(open_trades_log)
            priceable_count = len(daily_liability_itemization)
            unpriceable_count = len(unpriceable_positions)
            open_puts_tracker_sum = sum(open_puts_tracker.values())
            
            if priceable_count + unpriceable_count != open_trades_count:
                print(f"WARNING [DIAGNOSTIC] Position count mismatch on {date_str}:")
                print(f"    open_trades_log: {open_trades_count}")
                print(f"    priceable (in liability): {priceable_count}")
                print(f"    unpriceable (no ask): {unpriceable_count}")
                print(f"    priceable + unpriceable: {priceable_count + unpriceable_count}")
                print(f"    EXPECTED: All three should equal {open_trades_count}")
            
            if open_trades_count != open_puts_tracker_sum:
                print(f"WARNING [DIAGNOSTIC] Tracker mismatch on {date_str}:")
                print(f"    open_trades_log: {open_trades_count}")
                print(f"    open_puts_tracker sum: {open_puts_tracker_sum}")
                tracker_counts = {k: v for k, v in open_puts_tracker.items() if v > 0}
                print(f"    open_puts_tracker: {tracker_counts}")

        # Total Account Value (Net Asset Value)
        total_account_value = cash_balance - total_put_liability
        
        # Update peak NAV after computing total account value
        if total_account_value is not None:
            try:
                if float(total_account_value) > float(peak_account_value):
                    peak_account_value = float(total_account_value)
            except Exception:
                pass
        
        current_open_pos = sum(open_puts_tracker.values())
        if current_open_pos > peak_open_positions:
            peak_open_positions = current_open_pos
            
        # Update previous account value for next day's max premium calculation
        previous_account_value = total_account_value

        # Update monthly log
        realized_pnl_month, _, _ = monthly_pnl_log[month_key]
        spy_close_month = monthly_spy_prices.get(month_key, 0.0)
        monthly_pnl_log[month_key] = (realized_pnl_month + daily_pnl, total_account_value, spy_close_month)

        # Monthly progress logging
        runtime_str = time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - _sim_start_time))
        monthly_summary_msg = f"📅 {daily_date_obj}   Account Value: ${total_account_value:12,.2f}   RunTime: {runtime_str}"
        
        # Print monthly summary when month changes
        if last_printed_month and last_printed_month != month_key:
            _force_print(last_monthly_summary_msg)
        last_printed_month = month_key
        last_monthly_summary_msg = monthly_summary_msg

        # Detailed daily logging if minimal mode is off
        if not MINIMAL_PRINT_OUT:
            # Print Account Value breakdown (Corrected for Accuracy and Transparency)
            print(f"💵 **DAILY ACCOUNT VALUE (EOD - NAV):** ${total_account_value:,.2f}")
            print(f"  > **Cash Balance:** ${cash_balance:,.2f}")
            spy_current_price = 0.0
            if 'SPY' in stock_history_dict and date_str in stock_history_dict['SPY']:
                spy_current_price = stock_history_dict['SPY'][date_str].get('adj_close', 0.0)
            print(f" SPY current price: {spy_current_price}")
            print(f"  Open Puts: {sum(open_puts_tracker.values()):.2f}")
            
            # --- PROMOTED LIABILITY PRINT (This is the cumulative value) ---
            print(f"🛑 **TOTAL PORTFOLIO LIABILITY (Cost to Close):** ${total_put_liability:,.2f} (Computed using Ask Price)")
            
            # Print Itemized Liability Breakdown
            if daily_liability_itemization:
                for item in daily_liability_itemization:
                    print(item)
                    
            print(f"  > **Total accumulated Premium on Open Puts:** +${total_open_premium_collected:,.2f}")
            print(f"💰 **Cash Balance:** ${cash_balance:,.2f}")
            
            # Current Drawdown vs. peak NAV
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
                pass

            # Net Unrealized P&L
            print(f"  > **Net Unrealized P&L:** ${unrealized_pnl:,.2f}")
            
            # Print Realized P&L
            if daily_pnl != 0.0:
                print(f"💸 **DAILY NET REALIZED P&L:** ${daily_pnl:,.2f}")

            # Print cumulative realized P&L
            print(f"💰 **TOTAL NET REALIZED P&L (Cumulative):** ${cumulative_realized_pnl:,.2f}")

            # Print total P&L (Realized + Unrealized)
            print(f"💰 **TOTAL P&L (Realized + Unrealized):** ${(cumulative_realized_pnl + unrealized_pnl):,.2f}")

            # Print cash basis + total P&L (helpful sanity check)
            cash_plus_total_pnl = INITIAL_CASH + (cumulative_realized_pnl + unrealized_pnl)
            print(f"🧾 **INITIAL_CASH + TOTAL P&L (Cash Basis):** ${cash_plus_total_pnl:,.2f}")
            print(f"💵 **DAILY ACCOUNT VALUE (EOD - NAV):** ${total_account_value:,.2f}")
            print(f" SPY current price: {spy_current_price}")

            # Compare with NAV (total_account_value)
            try:
                # Allow a tiny numerical tolerance (1 cent)
                if abs(cash_plus_total_pnl - float(total_account_value)) <= 0.01:
                    print("✅ Total is the same as cash+gain")
                else:
                    print("ERROR Total is not the same as cash+gain")
                    print(f"  | cash_plus_total_pnl: ${cash_plus_total_pnl:,.2f}")
                    print(f"  | total_account_value: ${total_account_value:,.2f}")
            except Exception:
                print("WARNING Could not compare cash+gain to total_account_value due to an internal error.")
            
            # Add blank line after each day's processing
            print()
    
    # --- FINAL LIQUIDATION ---
    total_liquidation_pnl = 0.0
    if open_trades_log:
        for trade in open_trades_log:
            closing_ask = get_contract_exit_price(last_daily_orats_data, trade['ticker'], trade['expiration_date'], trade['strike']) or trade.get('last_known_ask', 0)
            qty = trade['quantity']
            premium_collected = trade['premium_received'] * qty * 100.0
            cost_to_close = closing_ask * qty * 100.0
            commission = qty * (COMMISSION_PER_CONTRACT + FINAL_COMMISSION_PER_CONTRACT)
            net_pnl = premium_collected - cost_to_close - commission
            total_liquidation_pnl += net_pnl
            # FIXED: Cash already holds premium, only subtract cost to close + commission
            cash_balance -= (cost_to_close + commission)
            closed_trades_count += 1
            if net_pnl > 0:
                winning_trades_count += 1
    
    final_nav = cash_balance
    
    # --- FINAL SUMMARY ---
    summary_text = []
    if sim_start_date and sim_end_date:
        total_days = (sim_end_date - sim_start_date).days
        total_years = total_days / 365.25 if total_days > 0 else 0
        annualized_gain_pct = ((final_nav / INITIAL_CASH) ** (1 / total_years) - 1) * 100 if total_years > 0 and final_nav > 0 else 0.0
        spy_annualized_return = ((spy_end_price / spy_start_price) ** (1 / total_years) - 1) * 100 if total_years > 0 and spy_start_price and spy_end_price else 0.0
        win_ratio = (winning_trades_count / closed_trades_count * 100) if closed_trades_count > 0 else 0.0
        
        summary_text.append("\n--- High-Level Performance Metrics ---")
        summary_text.append(f"| {'Metric':<31} | {'Value':<18} |")
        summary_text.append(f"|---------------------------------|--------------------|")
        summary_text.append(f"| {'Simulation Period':<31} | {total_days} days ({total_years:.2f} years) |")
        summary_text.append(f"| {'Final Account Value (NAV)':<31} | ${final_nav:17,.2f} |")
        summary_text.append(f"| {'Annualized Gain':<31} | {annualized_gain_pct:18.2f}% |")
        summary_text.append(f"| {'SPY Annualized Return':<31} | {spy_annualized_return:18.2f}% |")
        summary_text.append(f"| {'Worst Drawdown':<31} | {abs(worst_drawdown_pct * 100):18.2f}% |")
        summary_text.append(f"| {'Win Ratio':<31} | {win_ratio:18.2f}% |")
        summary_text.append(f"| {'Peak Open Positions':<31} | {peak_open_positions:<18} |")
        summary_text.append(f"|---------------------------------|--------------------|")
    else:
        annualized_gain_pct = 0.0
        summary_text.append("Simulation did not run for a sufficient period.")
    
    # Print the final summary to console/log
    _print("\n" + "\n".join(summary_text))
    _print("")  # Add extra newline
    
    # Add the Final Performance table that matches original format
    _print("REPORT Final Performance")
    _print("|----------------------------|-------------------------|")
    _print("| Parameter                  |  Value                  |")
    _print("|----------------------------|-------------------------|")
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    _print(f"| Current Date/Time          |        {current_datetime} |")
    _print(f"| Annualized Gain            |                  {annualized_gain_pct:.2f}% |")
    total_gain = final_nav - INITIAL_CASH
    _print(f"| Total Gain                 | $         {total_gain:,.2f} |")
    
    # Calculate runtime 
    elapsed_time = time.perf_counter() - _sim_start_time
    runtime_hours = int(elapsed_time // 3600)
    runtime_minutes = int((elapsed_time % 3600) // 60)
    runtime_seconds = int(elapsed_time % 60)
    runtime_str = f"{runtime_hours:02d}:{runtime_minutes:02d}:{runtime_seconds:02d}"
    _print(f"| Run Time                   |                {runtime_str} |")
    _print(f"| Peak Open Positions        |                      {peak_open_positions} |")
    
    # Count total entry events 
    total_entry_events = len([t for t in closed_trades_log]) + len(open_trades_log)
    _print(f"| Total Entry Events         |                     {total_entry_events} |")
    _print(f"| Win Ratio                  |                  {win_ratio:.2f}% |")
    
    # Get log filename (this could be passed in later if needed)
    log_filename = f"simulation_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    _print(f"| Log File                   |    {log_filename[:-21]}.log |")  # Simplified
    _print(f"| Worst Drawdown             |                 {abs(worst_drawdown_pct*100):.2f}% |")
    _print("|----------------------------|-------------------------|")
    _print("")
    
    return {
        "annualized_gain_pct": annualized_gain_pct,
        "worst_drawdown_pct": abs(worst_drawdown_pct * 100.0),
        "final_nav": final_nav,
        "full_summary_text": "\n".join(summary_text)
    }
