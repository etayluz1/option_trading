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
PARAM_SPECS = [
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
    (['exit_put_position', 'position_stop_loss_pct'], 'pct'),
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
    # Annualized Gain line: '| Annualized Gain            | {annualized_gain:>22.2f}% |'
    m1 = re.search(r"\|\s*Annualized Gain\s*\|\s*([+-]?\d+\.?\d*)%", text)
    m2 = re.search(r"\|\s*Worst Drawdown\s*\|\s*([+-]?\d+\.?\d*)%", text)
    annual = float(m1.group(1)) if m1 else None
    worst = float(m2.group(1)) if m2 else None
    return annual, worst


def append_result(record):
    line = orjson.dumps(record, option=orjson.OPT_SORT_KEYS).decode('utf-8')
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sweep optimizer for simulate.py rules")
    parser.add_argument("--baseline-only", action="store_true", help="Run only the baseline (original rules) and exit")
    parser.add_argument("--small-sweep", action="store_true", help="Use a smaller 3-step sweep (-25%, 0%, +25%)")
    parser.add_argument("--params", type=str, default=None, help="Comma-separated list of rule keys to sweep (e.g., entry_put_position.min_days_for_expiration,exit_put_position.position_stop_loss_pct)")
    parser.add_argument("--rules", type=str, default=None, help="Path to rules.json (defaults to workspace rules.json)")
    args = parser.parse_args()
    rules_path = os.path.abspath(args.rules) if args.rules else RULES_PATH
    if not os.path.exists(rules_path):
        print(f"rules.json not found at {rules_path}")
        print("Hint: pass --rules C:\\path\\to\\rules.json")
        return

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
