import os
import sys
import orjson
import time
import shutil
import subprocess
import re
import argparse
from datetime import datetime
from simulation_engine import run_simulation_in_memory, safe_percentage_to_float

# sweep_optimizer: one-factor-at-a-time sweeps over selected rules in rules.json,
# runs simulate.py for each modified rules file, parses the resulting log to extract
# Annualized Gain and Worst Drawdown, computes final_score = Annualized / Drawdown,
# and appends improvements to sweep_results.txt.

ROOT = os.path.abspath(os.path.dirname(__file__))
RULES_PATH = os.path.join(ROOT, "rules.json")
SIMULATE_SCRIPT = os.path.join(ROOT, "simulate.py")
LOGS_DIR = os.path.join(ROOT, "logs")
RESULTS_FILE = os.path.join(ROOT, "sweep_results.txt")
STOCK_HISTORY_PATH = os.path.join(ROOT, "stock_history.json")
ORATS_DIR = os.path.join(ROOT, "ORATS_json")

# Define which rule keys to sweep and their value kind. Path is a list of keys into rules dict.
# kind: 'int','float','dollar','pct' (percentage string like "5%")
# Exactly the requested rows in rules.json: 6,7,8; 12,13,14,15,16; 19-27; 35; 37
# Note: Row 8 in your list refers to max_puts_per_day (we are excluding the boolean Minimal_Print_Out).
PARAM_SPECS = [
    # Rows 6,7,8 (account_simulation)
    (['account_simulation', 'max_puts_per_account'], 'int'),
    (['account_simulation', 'max_puts_per_stock'], 'int'),
    (['account_simulation', 'max_puts_per_day'], 'int'),

    # Rows 12-16 (underlying_stock)
    (['underlying_stock', 'min_5_day_rise_pct'], 'pct'),
    (['underlying_stock', 'min_above_avg_pct'], 'pct'),
    (['underlying_stock', 'max_above_avg_pct'], 'pct'),
    (['underlying_stock', 'min_avg_up_slope_pct'], 'pct'),
    (['underlying_stock', 'min_stock_price'], 'dollar'),

    # Rows 19-27 (entry_put_position)
    (['entry_put_position', 'min_days_for_expiration'], 'int'),
    (['entry_put_position', 'max_days_for_expiration'], 'int'),
    (['entry_put_position', 'min_put_bid_price'], 'dollar'),
    (['entry_put_position', 'min_put_delta'], 'pct'),
    (['entry_put_position', 'max_put_delta'], 'pct'),
    (['entry_put_position', 'max_ask_above_bid_pct'], 'pct'),
    (['entry_put_position', 'min_avg_above_strike'], 'pct'),
    (['entry_put_position', 'min_risk_reward_ratio'], 'float'),
    (['entry_put_position', 'min_annual_risk_reward_ratio'], 'float'),
    (['entry_put_position', 'min_expected_profit'], 'pct'),

    # Rows 35, 37 (exit_put_position)
    (['exit_put_position', 'position_stop_loss_pct'], 'pct'),
    (['exit_put_position', 'stock_min_above_strike'], 'pct'),
]

SWEEP_STEPS = [-0.5, -0.375, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.375, 0.5]
SMALL_SWEEP_STEPS = [-0.25, 0.0, 0.25]


def load_rules(path):
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def save_rules(path, rules):
    with open(path, 'wb') as f:
        f.write(orjson.dumps(rules, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def get_by_path(d, keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def set_by_path(d, keys, value):
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def parse_value(raw, kind):
    if kind == 'int':
        return int(raw)
    if kind == 'float':
        return float(raw)
    if kind == 'dollar':
        # remove $ and commas
        return float(str(raw).replace('$', '').replace(',', '').strip())
    if kind == 'pct':
        # accept strings like "5%" or numeric 0.05
        if isinstance(raw, str) and '%' in raw:
            try:
                return float(raw.replace('%','').strip())/100.0
            except Exception:
                return float(raw)
        else:
            return float(raw)
    return raw


def format_value(val, kind):
    if kind == 'int':
        return int(round(val))
    if kind == 'float':
        return float(val)
    if kind == 'dollar':
        return f"${val:.2f}"
    if kind == 'pct':
        # val is decimal (0.05) -> format as '5%'
        return f"{val*100:.2f}%"
    return val


def load_data_cache():
    """Load stock history and prepare lazy-loading cache for ORATS data."""
    print("Loading stock history into memory cache...")
    cache = {
        "stock_history": None,
        "orats_data": {},  # Will be populated on-demand
        "_orats_loaded": set(),  # Track which dates have been loaded
        "_available_dates": []  # List of all available ORATS dates
    }

    # Load stock history
    if os.path.exists(STOCK_HISTORY_PATH):
        with open(STOCK_HISTORY_PATH, 'rb') as f:
            cache["stock_history"] = orjson.loads(f.read())
        print(f"Loaded stock history for {len(cache['stock_history'])} tickers.")
    else:
        print(f"Warning: stock_history.json not found at {STOCK_HISTORY_PATH}")
        return None

    # Get list of available ORATS dates without loading the files
    if not os.path.isdir(ORATS_DIR):
        print(f"Warning: ORATS_json directory not found at {ORATS_DIR}")
        return None
    
    json_files = sorted([f for f in os.listdir(ORATS_DIR) if f.endswith('.json')])
    cache["_available_dates"] = [f.split('.')[0] for f in json_files]
    print(f"Found {len(cache['_available_dates'])} ORATS files. Data will be loaded on-demand.")
    
    print("Data loading complete.")
    return cache


def lazy_load_orats_date(data_cache, date_str):
    """Load ORATS data for a specific date if not already loaded."""
    if date_str in data_cache["_orats_loaded"]:
        return  # Already loaded
    
    filepath = os.path.join(ORATS_DIR, f"{date_str}.json")
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data_cache["orats_data"][date_str] = orjson.loads(f.read())
        data_cache["_orats_loaded"].add(date_str)
    else:
        # Mark as loaded even if file doesn't exist to avoid repeated checks
        data_cache["_orats_loaded"].add(date_str)


def run_simulation_and_get_metrics(rules, data_cache):
    """Run simulation in-memory and return key metrics."""
    if not data_cache:
        print("Error: Data cache is not loaded.")
        return None, None, None, None, None

    # The engine now handles lazy loading internally
    # Pass the orats_data cache and the folder path
    summary = run_simulation_in_memory(
        rules, 
        data_cache["stock_history"], 
        all_orats_data=data_cache["orats_data"],
        orats_folder=ORATS_DIR
    )

    if not summary:
        return None, None, None, None, None

    # Extract metrics from the returned summary dictionary
    # Use correct keys that match what run_simulation_in_memory actually returns
    annual_gain = summary.get("annualized_gain_pct")
    worst_drawdown = summary.get("worst_drawdown_pct")
    final_nav = summary.get("final_nav")

    # The log file path might not be relevant anymore if not saved, but we can keep it for consistency
    log_path = summary.get("log_path", None)

    score = None
    if annual_gain is not None and worst_drawdown is not None and worst_drawdown != 0:
        score = annual_gain / worst_drawdown

    return annual_gain, worst_drawdown, score, final_nav, log_path


def find_logfile_from_stdout(stdout):
    m = re.search(r"Simulation complete\. Log saved to:\s*(.+)", stdout)
    if m:
        path = m.group(1).strip()
        # If relative, make absolute
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        return path
    # fallback: pick most recent file in logs dir
    if os.path.isdir(LOGS_DIR):
        files = [os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR) if os.path.isfile(os.path.join(LOGS_DIR, f))]
        if files:
            return max(files, key=os.path.getmtime)
    return None


def parse_metrics_from_log(logpath):
    """Open logfile and extract annualized gain (%) and worst drawdown (%) as floats."""
    if not logpath or not os.path.exists(logpath):
        return None, None
    text = open(logpath, 'r', encoding='utf-8', errors='ignore').read()
    # Primary summary table near the end
    m1 = re.search(r"\|\s*Annualized Gain\s*\|\s*([+-]?\d+\.?\d*)%", text)
    m2 = re.search(r"\|\s*Worst Drawdown\s*\|\s*([+-]?\d+\.?\d*)%", text)
    # Fallback: earlier comparison line with '**Annualized Gain (%)**'
    if not m1:
        m1 = re.search(r"\|\s*\*\*Annualized Gain \(%\)\*\*\s*\|\s*([+-]?\d+\.?\d*)%", text)
    annual = float(m1.group(1)) if m1 else None
    worst = float(m2.group(1)) if m2 else None

    # Fallback computation if summary not present: derive from daily logs
    if annual is None or worst is None:
        try:
            # Extract dates
            dates = re.findall(r"\n(\d{4}-\d{2}-\d{2})\n", text)
            if dates:
                from datetime import datetime as _dt
                first = _dt.strptime(dates[0], "%Y-%m-%d")
                last = _dt.strptime(dates[-1], "%Y-%m-%d")
                days = max(1, (last - first).days)
            else:
                days = None

            # Extract NAV values from cash-basis line
            navs = [float(v.replace(',', '')) for v in re.findall(r"INITIAL_CASH \+ TOTAL P&L \(Cash Basis\)\D+\$([0-9,]+\.[0-9]{2})", text)]
            if navs:
                initial_nav = navs[0]
                final_nav = navs[-1]
                if days and days > 0:
                    growth = final_nav / initial_nav if initial_nav > 0 else 1.0
                    annual_calc = (growth ** (365.0 / days) - 1.0) * 100.0
                    if annual is None:
                        annual = annual_calc

            # Extract worst drawdown as the minimum of Current Drawdown values (absolute percent)
            dd_vals = [float(m) for m in re.findall(r"Current Drawdown[\s\*:]*\s*([+-]?\d+\.?\d*)%", text)]
            if dd_vals:
                worst_dd = min(dd_vals)  # most negative
                if worst is None:
                    worst = abs(worst_dd)
        except Exception:
            pass

    return annual, worst


def extract_final_nav(logpath):
    """Return the last reported 'INITIAL_CASH + TOTAL P&L (Cash Basis): $X' value as float, or None."""
    if not logpath or not os.path.exists(logpath):
        return None
    try:
        text = open(logpath, 'r', encoding='utf-8', errors='ignore').read()
        navs = [float(v.replace(',', '')) for v in re.findall(r"INITIAL_CASH \+ TOTAL P&L \(Cash Basis\)\D+\$([0-9,]+\.[0-9]{2})", text)]
        if navs:
            return navs[-1]
    except Exception:
        return None
    return None


def _truncate(s, width=20):
    s = '' if s is None else str(s)
    return s if len(s) <= width else s[:max(0, width-1)] + '…'


def format_5cols_100(s1, s2, s3, s4, s5):
    """Return a single string of exactly 100 chars with five left-aligned fields
    starting at columns 1,21,41,61,81 (width 20 each).
    """
    w = 20
    s1 = _truncate(s1, w)
    s2 = _truncate(s2, w)
    s3 = _truncate(s3, w)
    s4 = _truncate(s4, w)
    s5 = _truncate(s5, w)
    line = f"{s1:<20}{s2:<20}{s3:<20}{s4:<20}{s5:<20}"
    return line[:100]


def print_run_line(label, param, value, final_nav, annual, worst, score):
    label_cap = (label or '').capitalize()
    ann_str = f"{annual:.2f}" if isinstance(annual, (int, float)) else "N/A"
    dd_str = f"{worst:.2f}" if isinstance(worst, (int, float)) else "N/A"
    sc_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
    val_str = f"${final_nav:,.0f}" if isinstance(final_nav, (int, float)) else "N/A"
    print(format_5cols_100(
        f"{label_cap}:",
        f"Val={val_str}",
        f"Ann:{ann_str}%",
        f"DD:{dd_str}%",
        f"Score={sc_str}"
    ))


def append_result(record):
    line = orjson.dumps(record, option=orjson.OPT_SORT_KEYS).decode('utf-8')
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def adjust_value(orig_val, kind, pct_change):
    """Return a new value adjusted by pct_change (e.g., 0.05 for +5%).
    Handles int/dollar/float/pct kinds. For 'pct' kind, pct_change applies to the underlying numeric value
    before formatting back to a percent string. Ensures sensible floors (e.g., ints >= 1).
    """
    if kind == 'int':
        delta = int(round(orig_val * pct_change))
        # ensure at least a movement of 1 when pct_change != 0
        if delta == 0 and pct_change != 0:
            delta = 1 if pct_change > 0 else -1
        new_val = orig_val + delta
        if new_val < 1:
            new_val = 1
        return new_val
    # Treat others as floats on the underlying numeric domain
    return orig_val * (1.0 + pct_change)


def optimize_param_hill(path_keys, kind, rules_path, data_cache, initial_step=0.05, expand_step=0.10, continue_step=0.05, max_iters=50, opt_start=None, opt_end=None):
    """
    Hill-climb optimizer for a single parameter.
    - Run baseline
    - Try +initial_step and -initial_step (these 3 runs cap the initial exploration)
    - If one improves, try a bigger move in that direction (expand_step)
    - If that improves, keep stepping in that direction by continue_step until no improvement
    Records baseline and any improvements to RESULTS_FILE. Restores original rules at end.
    Returns the best_record.
    """
    backup_path = rules_path + ".bak"
    shutil.copy2(rules_path, backup_path)
    print(f"Backed up original rules to {backup_path}")

    rules = load_rules(rules_path)
    orig_raw = get_by_path(rules, path_keys)
    if orig_raw is None:
        print(f"Parameter {'.'.join(path_keys)} not found in rules.json")
        return None
    orig_val = parse_value(orig_raw, kind)

    # Baseline
    print("Running baseline simulation with original rules...")
    rules_mod = load_rules(backup_path)
    if opt_start or opt_end:
        acct = rules_mod.setdefault('account_simulation', {})
        if opt_start:
            acct['start_date'] = opt_start
        if opt_end:
            acct['end_date'] = opt_end
    
    annual, worst, final_score, final_nav, logpath = run_simulation_and_get_metrics(rules_mod, data_cache)

    best_score = final_score if final_score is not None else -float('inf')
    baseline_value_disp = format_value(orig_val, kind)
    best_record = {
        'timestamp': datetime.now().isoformat(),
        'param': '.'.join(path_keys),
        'value': baseline_value_disp,
        'logfile': logpath,
        'annualized_gain_pct': annual,
        'worst_drawdown_pct': worst,
        'final_score': final_score,
        'final_nav': final_nav
    }
    append_result({'type': 'baseline', **best_record})
    print_run_line('baseline', best_record['param'], best_record['value'], final_nav, annual, worst, final_score)

    # Initial exploration: +5% and -5%
    def _run_with(val):
        rules_mod = load_rules(backup_path)
        set_by_path(rules_mod, path_keys, format_value(val, kind))
        if opt_start or opt_end:
            acct = rules_mod.setdefault('account_simulation', {})
            if opt_start:
                acct['start_date'] = opt_start
            if opt_end:
                acct['end_date'] = opt_end
        
        print(f"Running simulation for {'.'.join(path_keys)} = {format_value(val, kind)}...")
        a, w, sc, nav, lp = run_simulation_and_get_metrics(rules_mod, data_cache)
        return a, w, sc, nav, lp

    candidates = []
    for sign in (+1, -1):
        new_val = adjust_value(orig_val, kind, sign * initial_step)
        a, w, sc, nav, lp = _run_with(new_val)
        candidates.append((sign, new_val, a, w, sc, nav, lp))
        print_run_line('+' if sign > 0 else '-', '.'.join(path_keys), format_value(new_val, kind), nav, a, w, sc)

    # Pick best among baseline and the two neighbors
    direction = 0
    for sign, new_val, a, w, sc, nav, lp in candidates:
        if sc is not None and best_score is not None and sc > best_score:
            best_score = sc
            best_record = {
                'timestamp': datetime.now().isoformat(),
                'param': '.'.join(path_keys),
                'value': format_value(new_val, kind),
                'logfile': lp,
                'annualized_gain_pct': a,
                'worst_drawdown_pct': w,
                'final_score': sc,
                'final_nav': nav
            }
            append_result({'type': 'improvement', **best_record})
            print("New best after initial step:", best_record)
            direction = sign

    # If no improvement over baseline, stop for this param
    if direction == 0:
        shutil.copy2(backup_path, rules_path)
        print("Restored original rules.json from backup. No improvement for initial +/- step.")
        return best_record

    # Try a larger step in the improving direction
    iter_count = 0
    curr_val = parse_value(best_record['value'], kind)
    
    exp_val = adjust_value(orig_val, kind, direction * expand_step)
    a, w, sc, nav, lp = _run_with(exp_val)
    print_run_line('expand', '.'.join(path_keys), format_value(exp_val, kind), nav, a, w, sc)
    if sc is not None and sc > best_score:
        best_score = sc
        curr_val = exp_val
        best_record = {
            'timestamp': datetime.now().isoformat(),
            'param': '.'.join(path_keys),
            'value': format_value(curr_val, kind),
            'logfile': lp,
            'annualized_gain_pct': a,
            'worst_drawdown_pct': w,
            'final_score': sc,
            'final_nav': nav
        }
        append_result({'type': 'improvement', **best_record})
        print("Improved on expand step:", best_record)

    # Continue climbing in same direction with continue_step until no improvement
    while iter_count < max_iters:
        iter_count += 1
        next_val = adjust_value(curr_val, kind, direction * continue_step)
        a, w, sc, nav, lp = _run_with(next_val)
        print_run_line('step', '.'.join(path_keys), format_value(next_val, kind), nav, a, w, sc)
        if sc is not None and sc > best_score:
            best_score = sc
            curr_val = next_val
            best_record = {
                'timestamp': datetime.now().isoformat(),
                'param': '.'.join(path_keys),
                'value': format_value(curr_val, kind),
                'logfile': lp,
                'annualized_gain_pct': a,
                'worst_drawdown_pct': w,
                'final_score': sc,
                'final_nav': nav
            }
            append_result({'type': 'improvement', **best_record})
            print("Improved:", best_record)
        else:
            print("No further improvement. Stopping climb for this parameter.")
            break

    shutil.copy2(backup_path, rules_path)
    print("Restored original rules.json from backup.")
    return best_record


def main():
    parser = argparse.ArgumentParser(description="Sweep optimizer for simulate.py rules")
    parser.add_argument("--baseline-only", action="store_true", help="Run only the baseline (original rules) and exit")
    parser.add_argument("--small-sweep", action="store_true", help="Use a smaller 3-step sweep (-25%, 0%, +25%)")
    parser.add_argument("--params", type=str, default=None, help="Comma-separated list of rule keys to sweep (e.g., entry_put_position.min_days_for_expiration,exit_put_position.position_stop_loss_pct)")
    parser.add_argument("--rules", type=str, default=None, help="Path to rules.json (defaults to workspace rules.json)")
    parser.add_argument("--hill", action="store_true", help="Use hill-climbing optimization (baseline, +/-5%, then directional climb)")
    parser.add_argument("--triplet", action="store_true", help="Run exactly three runs for a parameter: -5%, 0%, +5% and report scores")
    parser.add_argument("--initial-step", type=float, default=5.0, help="Initial step size in percent for hill-climb (default 5.0)")
    parser.add_argument("--expand-step", type=float, default=10.0, help="Expand step size in percent when direction found (default 10.0)")
    parser.add_argument("--continue-step", type=float, default=5.0, help="Continue step size in percent for ongoing climb (default 5.0)")
    parser.add_argument("--opt-start", type=str, default=None, help="Optional override for account_simulation.start_date (e.g., 01/01/2025)")
    parser.add_argument("--opt-end", type=str, default=None, help="Optional override for account_simulation.end_date (e.g., 06/30/2025)")
    args = parser.parse_args()
    rules_path = os.path.abspath(args.rules) if args.rules else RULES_PATH
    if not os.path.exists(rules_path):
        print(f"rules.json not found at {rules_path}")
        print("Hint: pass --rules C:\\path\\to\\rules.json")
        return

    # Load all data into memory once at the beginning
    data_cache = load_data_cache()
    if not data_cache:
        print("Failed to load data cache. Exiting.")
        return

    if args.triplet:
        # Determine target parameter (default Rule 1)
        if args.params:
            wanted = args.params.split(',')[0].strip()
            key_parts = wanted.split('.') if wanted else ['account_simulation', 'max_puts_per_account']
        else:
            key_parts = ['account_simulation', 'max_puts_per_account']
        # find kind
        kind = None
        for keys, k in PARAM_SPECS:
            if keys == key_parts:
                kind = k
                break
        if kind is None:
            print(f"Unknown parameter {'.'.join(key_parts)} for triplet. Add it to PARAM_SPECS.")
            return

        # Backup original
        backup_path = rules_path + ".bak"
        shutil.copy2(rules_path, backup_path)
        print(f"Backed up original rules to {backup_path}")

        def _run_with_value(val):
            rules_mod = load_rules(backup_path)
            set_by_path(rules_mod, key_parts, format_value(val, kind))
            if args.opt_start or args.opt_end:
                acct = rules_mod.setdefault('account_simulation', {})
                if args.opt_start:
                    acct['start_date'] = args.opt_start
                if args.opt_end:
                    acct['end_date'] = args.opt_end
            
            disp = format_value(val, kind)
            print(f"Running simulation for {'.'.join(key_parts)} = {disp}...")
            annual, worst, score, final_nav, logpath = run_simulation_and_get_metrics(rules_mod, data_cache)
            return disp, logpath, annual, worst, score, final_nav

        # Baseline numeric value
        rules = load_rules(rules_path)
        orig_raw = get_by_path(rules, key_parts)
        orig_val = parse_value(orig_raw, kind)

        # Prepare sequence: -5%, 0%, +5%
        seq = [(-0.05, 'minus5'), (0.0, 'baseline'), (0.05, 'plus5')]
        results = []
        for pct, label in seq:
            if pct == 0.0:
                # Use original unmodified
                disp, logpath, annual, worst, score, final_nav = _run_with_value(orig_val)
            else:
                adj = adjust_value(orig_val, kind, pct)
                disp, logpath, annual, worst, score, final_nav = _run_with_value(adj)
            rec = {
                'timestamp': datetime.now().isoformat(),
                'mode': 'triplet',
                'label': label,
                'param': '.'.join(key_parts),
                'value': disp,
                'logfile': logpath,
                'annualized_gain_pct': annual,
                'worst_drawdown_pct': worst,
                'final_score': score,
                'final_nav': final_nav
            }
            append_result(rec)
            results.append(rec)

        # Restore original rules
        shutil.copy2(backup_path, rules_path)
        print("Restored original rules.json from backup.")

        # Pretty print concise report with requested format
        print("\n" + format_5cols_100("=== Triplet Results", "(-5%, 0%, +5%)", "", "", ""))
        best_rec = None
        baseline_rec = None
        for rec in results:
            label = str(rec.get('label', ''))
            label_cap = label.capitalize() if label else ''
            ann = rec.get('annualized_gain_pct')
            dd = rec.get('worst_drawdown_pct')
            ann_str = f"{ann:.2f}" if isinstance(ann, (int, float)) else "N/A"
            dd_str = f"{dd:.2f}" if isinstance(dd, (int, float)) else "N/A"
            final_nav = rec.get('final_nav')
            val_str = f"${final_nav:,.0f}" if isinstance(final_nav, (int, float)) else "N/A"
            sc = rec.get('final_score')
            sc_str = f"{sc:.2f}" if isinstance(sc, (int, float)) else "N/A"
            print(format_5cols_100(
                f"{label_cap}:",
                f"Val={val_str}",
                f"Ann:{ann_str}%",
                f"DD:{dd_str}%",
                f"Score={sc_str}"
            ))
            # Track best by score (higher is better)
            if label == 'baseline':
                baseline_rec = rec
            if isinstance(sc, (int, float)):
                if best_rec is None or sc > best_rec.get('final_score', float('-inf')):
                    best_rec = rec
        # Print recommendation if improvement over baseline
        if best_rec and baseline_rec:
            bsc = baseline_rec.get('final_score')
            if isinstance(best_rec.get('final_score'), (int, float)) and isinstance(bsc, (int, float)) and best_rec['final_score'] > bsc:
                print(format_5cols_100(
                    "Best:",
                    f"{best_rec['label'].capitalize()} improved",
                    f"New={best_rec['param']}",
                    f"Val={best_rec['value']}",
                    f"Score={best_rec['final_score']:.3f}"
                ))
            else:
                print(format_5cols_100("Best:", "No improvement", "over baseline", "in this triplet.", ""))
        return

    if args.hill:
        # Determine param to optimize; default to max_puts_per_account if not provided
        if args.params:
            wanted = args.params.split(',')[0].strip()
            key_parts = wanted.split('.') if wanted else ['account_simulation', 'max_puts_per_account']
        else:
            # Default to Rule 1: max_puts_per_account
            key_parts = ['account_simulation', 'max_puts_per_account']
        # find kind from PARAM_SPECS
        kind = None
        for keys, k in PARAM_SPECS:
            if keys == key_parts:
                kind = k
                break
        if kind is None:
            print(f"Unknown parameter {'.'.join(key_parts)} for hill-climb. Add it to PARAM_SPECS.")
            return
        # Run hill-climb for this param only
        optimize_param_hill(
            key_parts,
            kind,
            rules_path,
            data_cache,
            initial_step=args.initial_step / 100.0,
            expand_step=args.expand_step / 100.0,
            continue_step=args.continue_step / 100.0,
            opt_start=args.opt_start,
            opt_end=args.opt_end,
        )
        return

    # Legacy modes: baseline-only or sweeps
    # Backup original
    backup_path = rules_path + ".bak"
    shutil.copy2(rules_path, backup_path)
    print(f"Backed up original rules to {backup_path}")

    rules = load_rules(rules_path)

    # Baseline run (original rules)
    print("Running baseline simulation with original rules...")
    annual, worst, final_score, final_nav, logpath = run_simulation_and_get_metrics(rules, data_cache)

    best_score = final_score if final_score is not None else -float('inf')
    best_record = {
        'timestamp': datetime.now().isoformat(),
        'param': None,
        'value': None,
        'logfile': logpath,
        'annualized_gain_pct': annual,
        'worst_drawdown_pct': worst,
        'final_score': final_score,
        'final_nav': final_nav
    }
    append_result({'type': 'baseline', **best_record})
    print_run_line('baseline', '', '', final_nav, annual, worst, final_score)

    # Exit early if baseline-only requested
    if args.baseline_only:
        shutil.copy2(backup_path, rules_path)
        print("Restored original rules.json from backup.")
        print("Baseline-only run complete.")
        return

    # Sweep one parameter at a time
    # Filter params if --params provided
    selected_specs = PARAM_SPECS
    if args.params:
        wanted = {p.strip() for p in args.params.split(',') if p.strip()}
        def _join(kv):
            return '.'.join(kv)
        selected_specs = [spec for spec in PARAM_SPECS if _join(spec[0]) in wanted]
        if not selected_specs:
            print("No matching parameters for --params. Exiting.")
            shutil.copy2(backup_path, RULES_PATH)
            return

    steps = SMALL_SWEEP_STEPS if args.small_sweep else SWEEP_STEPS

    for path_keys, kind in selected_specs:
        orig_raw = get_by_path(rules, path_keys)
        if orig_raw is None:
            print(f"Skipping {'.'.join(path_keys)}: not found in rules.json")
            continue
        orig_val = parse_value(orig_raw, kind)
        print(f"Sweeping {'.'.join(path_keys)} (kind={kind}) starting from {orig_raw} -> parsed {orig_val}")

        for step in steps:
            # compute new value
            new_val = None
            if isinstance(orig_val, (int, float)):
                # For small original values (like days), use additive for int types
                if kind == 'int':
                    # if orig_val is small, step as +/- (max(1, int(orig*abs(step))))
                    delta = int(round(orig_val * step))
                    if delta == 0 and step != 0:
                        delta = 1 if step > 0 else -1
                    new_val = orig_val + delta
                    if new_val < 1:
                        new_val = 1
                else:
                    new_val = orig_val * (1.0 + step)
            else:
                # fallback
                try:
                    new_val = float(orig_val) * (1.0 + step)
                except Exception:
                    continue

            formatted = format_value(new_val, kind)

            # write modified rules
            rules_mod = load_rules(backup_path)  # start from original each time
            set_by_path(rules_mod, path_keys, formatted)
            save_rules(rules_path, rules_mod)

            print(f"Running simulation for {'.'.join(path_keys)} = {formatted}...")
            annual, worst, final_score, final_nav, logpath = run_simulation_and_get_metrics(rules_mod, data_cache)

            rec = {
                'timestamp': datetime.now().isoformat(),
                'param': '.'.join(path_keys),
                'value': formatted,
                'logfile': logpath,
                'annualized_gain_pct': annual,
                'worst_drawdown_pct': worst,
                'final_score': final_score,
                'final_nav': final_nav
            }

            print_run_line('result', rec['param'], rec['value'], final_nav, annual, worst, final_score)

            # If better than best (higher final_score), append to results file and update best
            try:
                if final_score is not None and best_score is not None:
                    # We're maximizing final_score (note: values may be negative)
                    if final_score > best_score:
                        best_score = final_score
                        append_result({'type': 'improvement', **rec})
                        print(f"New best score {best_score:.6f} for {rec['param']}={rec['value']}")
                elif final_score is not None and best_score is None:
                    best_score = final_score
                    append_result({'type': 'improvement', **rec})
            except Exception as e:
                print("Error comparing/appending result:", e)

    # restore original rules
    shutil.copy2(backup_path, rules_path)
    print("Restored original rules.json from backup.")
    print("Sweep complete.")


if __name__ == '__main__':
    main()
