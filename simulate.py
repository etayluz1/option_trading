import json
import os
from datetime import datetime
import math 

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
    total_contracts = 0
    
    # 1. Collect and sort all open positions
    for ticker, count in open_puts_tracker.items():
        if count > 0:
            open_tickers.append((ticker, count))
            total_contracts += count
            
    if not open_tickers:
        print("  (No open put positions.)")
        return
        
    open_tickers.sort(key=lambda x: x[0]) # Sort by ticker name
    
    # 2. Print the summary
    print(f"💼 **OPEN PORTFOLIO SUMMARY ({total_contracts} Total Contracts):**")
    
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
    

def load_and_run_simulation(rules_file_path, json_file_path):
    """
    Loads rules and data, initializes the tracker, and iterates chronologically 
    over ALL daily entries for ALL tickers starting from the specified date.
    Implements exit logic, calculates daily P&L and account value, respects 
    portfolio limits, and sells the optimal quantity based on premium collected.
    """
    
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
            
            print(f"✅ Simulation start date loaded: {start_date_str} (Parsed as {start_date_obj})")
            print(f"✅ All {len(rules['entry_put_position'])} Entry Rules loaded successfully.")
            print(f"✅ Account Limits: Max Puts Total: {MAX_PUTS_PER_ACCOUNT}, Max Puts Per Stock: {MAX_PUTS_PER_STOCK}")
            print(f"✅ Trading Cost: Commission per contract is ${COMMISSION_PER_CONTRACT:.2f}")
            print(f"✅ Stop Loss Rule (SMA150): Max Stock Drop Below SMA150 = {STOCK_MAX_BELOW_AVG_PCT * 100:.2f}%")


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
    
    # --- Financial Trackers ---
    cash_balance = INITIAL_CASH # Tracks cash directly
    cumulative_realized_pnl = 0.0 # Tracks profit from closed trades
    
    # --- Performance Tracking ---
    sim_start_date = None
    sim_end_date = None
    spy_start_price = None
    spy_end_price = None
    
    # NEW: Monthly P&L Log (Tuple: (Realized PNL for month, EOD Total Value))
    # EOD Total Value is the Mark-to-Market value (Cash + Unrealized PNL)
    monthly_pnl_log = {} 
    
    # NEW: Exit Counters
    stop_loss_count = 0
    expired_otm_count = 0
    expired_itm_count = 0

    all_tickers = list(open_puts_tracker.keys())
    print(f"✅ Trackers initialized for {len(all_tickers)} tickers.")
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
    
    # Set for quick checking of active positions (Ticker, Strike, Expiration)
    active_position_keys = set()
    
    # Variable to hold last day's ORATS data for final liquidation (if needed)
    last_daily_orats_data = None 
    
    for date_str in sorted_unique_dates:
        daily_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        if daily_date_obj >= start_date_obj:
            
            # --- START DAILY PROCESSING ---
            # Capture simulation dates and SPY prices
            if sim_start_date is None:
                sim_start_date = daily_date_obj
                if 'SPY' in stock_history_dict and date_str in stock_history_dict['SPY']:
                    spy_start_price = stock_history_dict['SPY'][date_str].get('adj_close')
            
            sim_end_date = daily_date_obj # Update end date every successful day
            if 'SPY' in stock_history_dict and date_str in stock_history_dict['SPY']:
                spy_end_price = stock_history_dict['SPY'][date_str].get('adj_close')
            
            
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
            
            current_account_puts = sum(open_puts_tracker.values())
            daily_pnl = 0.0 # Realized P&L from closed trades today
            
            # --- Liability Trackers (Re-initialized daily for fresh MTM calculation) ---
            unrealized_pnl = 0.0 
            total_put_liability = 0.0 
            total_open_premium_collected = 0.0 
            
            # FIX 1: Initialize total_account_value and account_full_today at the start of the loop
            total_account_value = cash_balance # Start with last known cash value as a base
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
                    
                    # --- Calculate Exit P&L ---
                    exit_commission = qty * COMMISSION_PER_CONTRACT
                    premium_collected_gross = trade['premium_received'] * qty * 100.0
                    
                    if expired_triggered:
                        is_itm = current_adj_close < trade['strike']
                        
                        if is_itm:
                            payout_gross = (trade['strike'] - current_adj_close) * qty * 100.0
                            cost_to_close_gross = payout_gross 
                            net_profit = premium_collected_gross - cost_to_close_gross - exit_commission
                            expired_itm_count += qty
                            exit_type = "Expiration (ITM/Assigned)"
                        else:
                            cost_to_close_gross = 0.0
                            net_profit = premium_collected_gross - exit_commission
                            expired_otm_count += qty
                            exit_type = "Expiration (OTM/Max Profit)"
                        
                    elif stop_loss_triggered and current_ask_price is not None:
                        # Close at current conservative Ask Price (Loss or small Gain)
                        cost_to_close_gross = current_ask_price * qty * 100.0
                        net_profit = premium_collected_gross - cost_to_close_gross - exit_commission
                        stop_loss_count += qty
                        exit_type = f"STOP LOSS (Stock Below SMA150)"
                    else:
                        # Cannot determine exit price for stop-loss, skip for now 
                        continue 

                    daily_pnl += net_profit
                    
                    # Cash change = (Premium Collected) - (Cost to Close) - (Exit Commission)
                    cash_change = net_profit 
                    cash_balance += cash_change
                    
                    # Log the exit
                    print(f"🔥 **EXIT:** {exit_type}: {trade['ticker']} (Strike ${trade['strike']:.2f}, Qty {qty}). Net Profit: ${net_profit:,.2f}")
                    
                    # Update trackers
                    positions_to_remove.append(i)
                    open_puts_tracker[trade['ticker']] -= qty 
                    active_position_keys.remove(trade['unique_key'])
            
            # --- Monthly and Yearly P&L Aggregation (Start of Day P&L aggregation) ---
            month_key = (daily_date_obj.year, daily_date_obj.month)
            
            if month_key not in monthly_pnl_log:
                # Value is: (Realized PNL for month, Final Value)
                monthly_pnl_log[month_key] = (daily_pnl, 0.0) 
            else:
                current_pnl, _ = monthly_pnl_log[month_key]
                monthly_pnl_log[month_key] = (current_pnl + daily_pnl, 0.0) 
            
            # Update cumulative P&L with today's realized profit (Net of exit commissions)
            cumulative_realized_pnl += daily_pnl
            
            # Remove closed positions from the log, iterating backwards
            for index in sorted(positions_to_remove, reverse=True):
                open_trades_log.pop(index)
            
            # Recalculate current_account_puts after exits
            current_account_puts = sum(open_puts_tracker.values())


            # ----------------------------------------------
            # --- Market Scan and Trade Entry ---
            # ----------------------------------------------
            
            # Check if we should skip the market scan due to global limits
            if current_account_puts >= MAX_PUTS_PER_ACCOUNT and not account_full_today:
                print(f"\n>>>> Date: {date_str} (Investable Tickers) <<<<")
                print(f"🛑 **ACCOUNT FULL (Global Limit):** {current_account_puts}/{MAX_PUTS_PER_ACCOUNT} contracts. Skipping scan for new trades.")
                account_full_today = True
            
            # Print the daily header and limits info for scan days
            if not account_full_today:
                print(f"\n>>>> Date: {date_str} (Investable Tickers) <<<<")
                print(f"📈 **Max Premium per Trade:** ${MAX_PREMIUM_PER_TRADE:,.2f}")

            if not account_full_today:
                
                # Inner loop: Check ALL tickers for viable contracts
                for ticker in all_tickers:
                    # Check 1 & 2: Data exists for date AND ticker is investable
                    if date_str in stock_history_dict[ticker] and stock_history_dict[ticker][date_str].get('investable') is True:
                        
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
                                            passes_risk_reward = False
                                            risk_reward_ratio = None
                                            
                                            if pbidpx_value > 0 and strike_value > pbidpx_value:
                                                risk_reward_ratio = calculate_risk_reward_ratio(strike_value, pbidpx_value)
                                                if risk_reward_ratio is not None:
                                                    passes_risk_reward = risk_reward_ratio > MIN_RISK_REWARD_RATIO
                                            
                                            if passes_risk_reward:
                                                # Store R/R ratio and Adj. Close on the option for easy access
                                                option['calculated_rr_ratio'] = risk_reward_ratio
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
                ask_at_entry = 0.0 

                if daily_trade_candidates:
                    # Sort the ENTIRE list of candidates globally by R/R ratio (highest R/R first)
                    daily_trade_candidates.sort(key=lambda x: x.get('calculated_rr_ratio', -float('inf')), reverse=True)
                    
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
                            
                            # FIX: Ensure pBidPx is a float, handling potential string issues
                            try:
                                pBidPx_value = float(contract['pBidPx'])
                                paskPx_value = float(contract['pAskPx']) # Get Ask for logging
                            except ValueError:
                                # Skip the contract if the bid price is invalid
                                continue
                                
                            premium_per_contract = pBidPx_value * 100.0
                            
                            # Calculate quantity based on max premium per trade (floored)
                            if premium_per_contract > 0:
                                qty_by_premium = math.floor(MAX_PREMIUM_PER_TRADE / premium_per_contract)
                            else:
                                qty_by_premium = 0

                            # Determine quantity based on remaining contract limits
                            remaining_account_slots = MAX_PUTS_PER_ACCOUNT - current_account_puts
                            remaining_stock_slots = MAX_PUTS_PER_STOCK - open_puts_tracker[ticker_check]
                            
                            # Final quantity is the minimum of the three constraints
                            trade_quantity = min(qty_by_premium, remaining_account_slots, remaining_stock_slots)
                            
                            if trade_quantity >= 1:
                                best_contract = contract
                                ask_at_entry = paskPx_value # Capture Ask Price
                                bid_at_entry = pBidPx_value # Capture Bid Price
                                break # Found the best eligible contract with trade quantity >= 1
                        
                    
                    if best_contract:
                        
                        print(f"🥇 **ABSOLUTE BEST CONTRACT TODAY (Ranked by R/R Ratio):**")
                        
                        # Fetch the original delta value using the safer function
                        original_delta = best_contract.get('putDelta')
                        delta_float = safe_percentage_to_float(original_delta)
                        delta_str = f"{delta_float:.4f}" if delta_float is not None else "N/A"
                        
                        # Re-calculate values for printing
                        total_premium_collected = premium_per_contract * trade_quantity

                        best_info = (
                            f"  1. **{best_contract['ticker']}:** Qty={trade_quantity}, "
                            f"Total Premium Collected=${total_premium_collected:,.2f}, "
                            f"Strike=${best_contract['strike']:.2f}, "
                            f"DTE={best_contract['dte']}, "
                            f"R/R={best_contract['calculated_rr_ratio']:.2f}"
                        )
                        print(best_info)
                        
                    else:
                        print("❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (All candidates failed limits/duplication checks or resulted in Qty=0).")
                        
                else:
                    print(f"❌ **ABSOLUTE BEST CONTRACT TODAY:** None found across all tickers (No contract passed filters).")
                
                
                # --- TRADING LOGIC: ENTER POSITION ---
                if best_contract and trade_quantity >= 1:
                    ticker_to_enter = best_contract['ticker']
                    
                    # 1. Commission Cost (Realized Loss)
                    entry_commission = trade_quantity * COMMISSION_PER_CONTRACT
                    
                    # 2. Calculate trade P&L (Realized Loss: Commission Only)
                    daily_pnl -= entry_commission
                    cumulative_realized_pnl -= entry_commission
                    
                    # 3. Update cash balance (Inflow of Premium - Commission)
                    premium_inflow = premium_per_contract * trade_quantity
                    cash_balance += premium_inflow
                    cash_balance -= entry_commission 
                    
                    # 4. Update the position count
                    open_puts_tracker[ticker_to_enter] += trade_quantity
                    
                    # 5. Log the trade details (include quantity)
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
                    
                    # 6. Update the quick-check set
                    active_position_keys.add(trade_entry['unique_key'])
                    
                    # 7. Print the consolidated portfolio summary
                    print_daily_portfolio_summary(open_puts_tracker)
                    
                    # --- NEW: DETAILED TRANSACTION LOG ---
                    position_liability_at_entry = ask_at_entry * trade_quantity * 100.0
                    
                    print("\n📈 **TODAY'S ENTRY TRANSACTION DETAILS:**")
                    print(f"  | Ticker/Contract: {ticker_to_enter} (Qty {trade_quantity})")
                    print(f"  | Bid Price: ${bid_at_entry:.2f} | Ask Price: ${ask_at_entry:.2f}")
                    print(f"  | Cash Inflow (Gross Premium): +${premium_inflow:,.2f}")
                    print(f"  | Commission Cost (Realized P&L): -${entry_commission:,.2f}")
                    print(f"  | Cash Balance (After Trade): ${cash_balance:,.2f}")
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

            # Total Account Value = Cash Balance + Net Unrealized P&L 
            total_account_value = cash_balance + unrealized_pnl
            
            # FIX 2: Update the monthly log with the EOD Total Account Value for the last trading day of the month.
            month_key = (daily_date_obj.year, daily_date_obj.month)
            current_pnl, _ = monthly_pnl_log.get(month_key, (0.0, 0.0))
            # Only update the EOD value if this is the last day of the month (or the last day of the sim)
            # Simplest approach: overwrite the value every day, ensuring the last day's MTM value is captured.
            monthly_pnl_log[month_key] = (current_pnl, total_account_value) 
            
            # Print Account Value breakdown (Corrected for Accuracy and Transparency)
            print(f"💵 **DAILY ACCOUNT VALUE (EOD):** ${total_account_value:,.2f}")
            
            # --- PROMOTED LIABILITY PRINT (This is the cumulative value) ---
            print(f"🛑 **TOTAL PORTFOLIO LIABILITY (Cost to Close):** ${total_put_liability:,.2f} (Computed using Ask Price)")
            
            # Print Itemized Liability Breakdown
            if daily_liability_itemization:
                for item in daily_liability_itemization:
                    print(item)

            print(f"  > **Cash Balance:** ${cash_balance:,.2f}")
            print(f"  > **Total Premium on Open Puts:** +${total_open_premium_collected:,.2f}")
            print(f"  > **Net Unrealized P&L:** ${unrealized_pnl:,.2f}")
            
            # Print Realized P&L
            if daily_pnl != 0.0:
                print(f"💸 **DAILY NET REALIZED P&L:** ${daily_pnl:,.2f}")
                
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
                premium_collected_gross = premium_collected_per_contract * qty * 100.0
                cost_to_close_gross = closing_ask * qty * 100.0
                exit_commission = qty * FINAL_COMMISSION_PER_CONTRACT
                
                # P&L Calculation: (Initial Premium) - (Cost to Close) - (Exit Commission)
                position_net_gain = premium_collected_gross - cost_to_close_gross - exit_commission
                
                total_liquidation_pnl += position_net_gain
                
                # Adjust cash balance: Cash balance already holds the premium. Now we pay the cost to close.
                cash_balance -= cost_to_close_gross
                cash_balance -= exit_commission
                
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
        print(f"💰 **FINAL ACCOUNT VALUE (CASH):** ${final_account_value_liquidated:,.2f}")
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
    
    print("\n| Metric                  | Portfolio Gain    | SPY Benchmark  | Comparison       |")
    print("|-------------------------|-------------------|----------------|------------------|")
    # FIX 4: Aligned columns using refined explicit width and right alignment (>)
    # Portfolio Gain (16), SPY Benchmark (13), Comparison (10)
    print(f"| **Total Net Gain (%)** | {percent_total_gain:>16.2f}% | {spy_total_return:>13.2f}% | **{percent_total_gain - spy_total_return:>10.2f}pp** |")
    print(f"| **Annualized Gain (%)** | {annualized_gain:>16.2f}% | {spy_annualized_return:>13.2f}% | **{annualized_gain - spy_annualized_return:>10.2f}pp** |")
    
    # 8. Monthly and Yearly Performance Tables
    
    # --- P&L Aggregation for Tables ---
    monthly_performance = {} # Final structure: (Month) -> (% Gain, $ Gain, End Value)
    yearly_performance = {}
    
    # Pre-calculate the starting value for each month
    sorted_months = sorted(monthly_pnl_log.keys())
    
    # --- Monthly Calculation Pass ---
    for i, (year, month) in enumerate(sorted_months):
        pnl_realized, month_end_value_mtm = monthly_pnl_log[(year, month)]
        
        # Determine Base Value (Start MTM Value)
        if i == 0:
            month_start_value = INITIAL_CASH
        else:
            # FIX 5: Calculate the start value based on the previous month's END MTM value
            prev_year, prev_month = sorted_months[i-1]
            # Retrieve the previous month's MTM value (index 1)
            _, prev_eod_value_mtm = monthly_pnl_log[(prev_year, prev_month)] 
            month_start_value = prev_eod_value_mtm
            
        # ***CRITICAL FIX***: Replace the last month's MTM value with the final liquidiated cash value for a clean report closure.
        # This resolves the discrepancy pointed out by the user between the final monthly entry and the Final Liquidation Summary.
        is_last_month = i == len(sorted_months) - 1
        month_end_value_reported = final_account_value_liquidated if is_last_month else month_end_value_mtm
        
        # Calculate Monthly Metrics
        # NOTE: Using month_end_value_reported to calculate gain vs previous month's MTM start value
        monthly_gain_abs = month_end_value_reported - month_start_value
        monthly_gain_pct = (monthly_gain_abs / month_start_value) * 100.0 if month_start_value > 0 else 0.0
        
        monthly_performance[(year, month)] = (monthly_gain_pct, monthly_gain_abs, month_end_value_reported)
        
        # Aggregate to Year (Yearly Start Value = MTM at start of year; Yearly End Value = MTM/Realized at end of year)
        if year not in yearly_performance:
            yearly_performance[year] = {
                'start_value': month_start_value, # Start MTM value of the first month in the year (or sim)
                'end_value': month_end_value_reported,
                'realized_pnl': pnl_realized
            }
        else:
            # Update the End Value to the current month's reported end value
            yearly_performance[year]['end_value'] = month_end_value_reported
            yearly_performance[year]['realized_pnl'] += pnl_realized

    # --- Print Monthly Table ---
    print("")
    print("\n--- MONTHLY PORTFOLIO GAIN ---")
    print("| Month   | Total Value End | $ Gain       | % Gain  |")
    print("|---------|-----------------|--------------|---------|") # Fix 9: Aligned separator dashes
    
    for (year, month), (pct_gain, abs_gain, end_value) in monthly_performance.items():
        month_label = datetime(year, month, 1).strftime('%Y-%m')
        
        # FIX 6: Aligned columns using refined explicit width and right alignment (>)
        # Total Value End (15) , $ Gain (12), % Gain (8)
        # Split $ and number to align pipes
        print(f"| {month_label:^5} | $ {end_value:>11,.2f}   | $ {abs_gain:>9,.2f} | {pct_gain:>6.2f}% |")

    # --- Print Yearly Table ---
    print("")
    print("\n--- YEARLY PORTFOLIO GAIN ---")
    print("| Year    | Total Value End | $ Gain       | % Gain  |")
    print("|---------|-----------------|--------------|---------|") # Fix 9: Aligned separator dashes
    
    for year in sorted(yearly_performance.keys()):
        data = yearly_performance[year]
        year_end_value = data['end_value']
        year_start_value = data['start_value']
        
        yearly_gain_abs = year_end_value - year_start_value
        yearly_gain_pct = (yearly_gain_abs / year_start_value) * 100.0 if year_start_value > 0 else 0.0
        
        # FIX 7: Aligned columns using refined explicit width and right alignment (>)
        # Total Value End (15) , $ Gain (12), % Gain (8)
        # Split $ and number to align pipes
        print(f"| {year:^5}   | $ {year_end_value:>11,.2f}   | $ {yearly_gain_abs:>9,.2f} | {yearly_gain_pct:>6.2f}% |")

    # 9. Exit Statistics
    total_closed_positions = stop_loss_count + expired_otm_count + expired_itm_count
    
    print("")
    print("\n--- TRADE EXIT STATISTICS (by Contract Count) ---")
    print("| Exit Reason                  | Contracts Closed  | % of Total Closed  |")
    print("|------------------------------|-------------------|--------------------|")
    # FIX 8: Aligned columns using refined explicit width and right alignment (>)
    # Contracts Closed (16), % of Total Closed (17)
    print(f"| **Stop Loss** | {stop_loss_count:>16,}  | {stop_loss_count / total_closed_positions * 100 if total_closed_positions > 0 else 0:>17.2f}% |")
    print(f"| **Expired OTM (Max Profit)** | {expired_otm_count:>16,}  | {expired_otm_count / total_closed_positions * 100 if total_closed_positions > 0 else 0:>17.2f}% |")
    print(f"| **Expired ITM (Assignment)** | {expired_itm_count:>16,}  | {expired_itm_count / total_closed_positions * 100 if total_closed_positions > 0 else 0:>17.2f}% |")
    print(f"| **Total Positions Closed** | {total_closed_positions:>16,}  | {100.0:>17.2f}% |")
    
# Execute the main function
load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)