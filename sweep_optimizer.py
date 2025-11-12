import os
import sys
import orjson
import time
import shutil
import subprocess
import re
import argparse
from datetime import datetime

# sweep_optimizer: one-factor-at-a-time sweeps over selected rules in rules.json,
# runs simulate.py for each modified rules file, parses the resulting log to extract
# Annualized Gain and Worst Drawdown, computes final_score = Annualized / Drawdown,
# and appends improvements to sweep_results.txt.

ROOT = os.path.abspath(os.path.dirname(__file__))
RULES_PATH = os.path.join(ROOT, "rules.json")
SIMULATE_SCRIPT = os.path.join(ROOT, "simulate.py")
LOGS_DIR = os.path.join(ROOT, "logs")
RESULTS_FILE = os.path.join(ROOT, "sweep_results.txt")

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


def run_simulation():
    """Run simulate.py and return (stdout, stderr, returncode)."""
    proc = subprocess.run([sys.executable, SIMULATE_SCRIPT], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout, proc.stderr, proc.returncode


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
    return annual, worst


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


def optimize_param_hill(path_keys, kind, rules_path, initial_step=0.05, expand_step=0.10, continue_step=0.05, max_iters=50, opt_start=None, opt_end=None):
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
    # If date overrides are provided, write a temp rules file reflecting them before baseline
    rules_mod = load_rules(backup_path)
    if opt_start or opt_end:
        acct = rules_mod.setdefault('account_simulation', {})
        if opt_start:
            acct['start_date'] = opt_start
        if opt_end:
            acct['end_date'] = opt_end
        save_rules(rules_path, rules_mod)
    else:
        # Ensure we start from original rules
        shutil.copy2(backup_path, rules_path)
    stdout, stderr, rc = run_simulation()
    logpath = find_logfile_from_stdout(stdout)
    annual, worst = parse_metrics_from_log(logpath)
    final_score = None
    if annual is not None and worst is not None and worst != 0:
        final_score = annual / worst
    best_score = final_score if final_score is not None else -float('inf')
    best_record = {
        'timestamp': datetime.now().isoformat(),
        'param': None,
        'value': orig_raw,
        'logfile': logpath,
        'annualized_gain_pct': annual,
        'worst_drawdown_pct': worst,
        'final_score': final_score
    }
    append_result({'type': 'baseline', **best_record})
    print("Baseline:", best_record)

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
        save_rules(rules_path, rules_mod)
        print(f"Running simulation for {'.'.join(path_keys)} = {format_value(val, kind)}...")
        stdout, stderr, rc = run_simulation()
        lp = find_logfile_from_stdout(stdout)
        a, w = parse_metrics_from_log(lp)
        sc = None
        if a is not None and w is not None and w != 0:
            sc = a / w
        return a, w, sc, lp

    candidates = []
    for sign in (+1, -1):
        new_val = adjust_value(orig_val, kind, sign * initial_step)
        a, w, sc, lp = _run_with(new_val)
        candidates.append((sign, new_val, a, w, sc, lp))

    # Pick best among baseline and the two neighbors
    direction = 0
    for sign, new_val, a, w, sc, lp in candidates:
        if sc is not None and best_score is not None and sc > best_score:
            best_score = sc
            best_record = {
                'timestamp': datetime.now().isoformat(),
                'param': '.'.join(path_keys),
                'value': format_value(new_val, kind),
                'logfile': lp,
                'annualized_gain_pct': a,
                'worst_drawdown_pct': w,
                'final_score': sc
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
    a, w, sc, lp = _run_with(adjust_value(orig_val, kind, direction * expand_step))
    if sc is not None and sc > best_score:
        best_score = sc
        curr_val = adjust_value(orig_val, kind, direction * expand_step)
        best_record = {
            'timestamp': datetime.now().isoformat(),
            'param': '.'.join(path_keys),
            'value': format_value(curr_val, kind),
            'logfile': lp,
            'annualized_gain_pct': a,
            'worst_drawdown_pct': w,
            'final_score': sc
        }
        append_result({'type': 'improvement', **best_record})
        print("Improved on expand step:", best_record)

    # Continue climbing in same direction with continue_step until no improvement
    while iter_count < max_iters:
        iter_count += 1
        next_val = adjust_value(curr_val, kind, direction * continue_step)
        a, w, sc, lp = _run_with(next_val)
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
                'final_score': sc
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
    stdout, stderr, rc = run_simulation()
    logpath = find_logfile_from_stdout(stdout)
    annual, worst = parse_metrics_from_log(logpath)
    final_score = None
    if annual is not None and worst is not None and worst != 0:
        final_score = annual / worst

    best_score = final_score if final_score is not None else -float('inf')
    best_record = {
        'timestamp': datetime.now().isoformat(),
        'param': None,
        'value': None,
        'logfile': logpath,
        'annualized_gain_pct': annual,
        'worst_drawdown_pct': worst,
        'final_score': final_score
    }
    print("Baseline result:", best_record)
    append_result({'type': 'baseline', **best_record})

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
            stdout, stderr, rc = run_simulation()
            logpath = find_logfile_from_stdout(stdout)
            annual, worst = parse_metrics_from_log(logpath)
            final_score = None
            if annual is not None and worst is not None and worst != 0:
                final_score = annual / worst

            rec = {
                'timestamp': datetime.now().isoformat(),
                'param': '.'.join(path_keys),
                'value': formatted,
                'logfile': logpath,
                'annualized_gain_pct': annual,
                'worst_drawdown_pct': worst,
                'final_score': final_score
            }

            print('Result:', rec)

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
