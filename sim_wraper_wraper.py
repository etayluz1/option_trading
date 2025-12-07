import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parent
SIMULATOR_PATH = ROOT_DIR / "simulate_yuda.py"
LOGS_DIR = ROOT_DIR / "logs"
RULES_PATH = ROOT_DIR / "rules.json"

# Match the console line emitted by simulate_yuda.py with the generated log path.
LOG_LINE_RE = re.compile(r"Simulation complete\. Log saved to:\s*(.+)")
# Capture the final NAV line, allowing for both "FINAL REALIZED CASH VALUE" and
# the older double-word variant "FINAL ACCOUNT ACCOUNT VALUE (CASH)".
FINAL_NAV_RE = re.compile(
    r"FINAL (?:REALIZED CASH|ACCOUNT(?: ACCOUNT)? VALUE(?: \([^)]+\))?):\s*\$\s*([\d,]+(?:\.\d+)?)"
)
# Fallback to the last daily NAV line if the final summary could not be parsed.
DAILY_NAV_RE = re.compile(r"DAILY ACCOUNT VALUE \(EOD - NAV\):\s*\$\s*([\d,]+(?:\.\d+)?)")
# Capture the total gain line from the final liquidation summary.
TOTAL_GAIN_RE = re.compile(r"TOTAL NET PROFIT \(Start to Finish\):\s*\$\s*([\d,]+(?:\.\d+)?)")
# Fallback to the cumulative realized P&L line if TOTAL NET PROFIT is absent.
CUM_REALIZED_RE = re.compile(r"TOTAL NET REALIZED P&L \(Cumulative\):\s*\$\s*([\d,]+(?:\.\d+)?)")
# Capture annualized gain percentages from the final performance table.
ANNUALIZED_GAIN_RE = re.compile(r"Annualized Gain[^%]*?([0-9]+(?:\.[0-9]+)?)%")


TOTAL_GAIN_RE = re.compile(r"TOTAL NET PROFIT \(Start to Finish\):\s*\$\s*([\d,]+(?:\.[\d]+)?)")
# Capture the Total Gain value from the summary table (monitor output)
TOTAL_GAIN_TABLE_RE = re.compile(r"Total Gain\s*\|\s*\$\s*([\d,]+(?:\.[\d]+)?)\s*\|")

# wrapper_sweep_pct_set = [0.3, 2, 3, 5, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 50]  # Percentages
wrapper_sweep_pct_set = [0.3, 2]  # Percentages
score_improvements_count = 0

def _current_log_names() -> set[str]:
    if not LOGS_DIR.exists():
        return set()
    return {
        path.name
        for path in LOGS_DIR.iterdir()
        if path.is_file()
    }


def _latest_log_file() -> Optional[Path]:
    if not LOGS_DIR.exists():
        return None
    log_files = [path for path in LOGS_DIR.glob("*.log") if path.is_file()]
    if not log_files:
        return None
    return max(log_files, key=lambda p: p.stat().st_mtime)


def _resolve_log_path(raw_path: str) -> Path:
    candidate = Path(raw_path.strip().strip('"'))
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


def _extract_float(patterns: Iterable[re.Pattern[str]], text: str) -> Optional[float]:
    for pattern in patterns:
        matches = pattern.findall(text)
        if matches:
            value = matches[-1].replace(",", "")
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _format_money(value: Optional[float]) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def _format_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}%"


def _format_score(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _format_runtime(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d} ({seconds:.2f}s)"


def _locate_log_file(stdout: str, before: set[str], after: set[str]) -> Path:
    resolved: Optional[Path] = None
    for line in stdout.splitlines():
        match = LOG_LINE_RE.search(line)
        if match:
            candidate = _resolve_log_path(match.group(1))
            if candidate.exists():
                resolved = candidate
    if resolved:
        return resolved

    new_names = sorted(after - before)
    if new_names:
        candidate = LOGS_DIR / new_names[-1]
        if candidate.exists():
            return candidate

    latest = _latest_log_file()
    if latest:
        return latest

    raise RuntimeError("Simulation completed but the log file could not be located.")


def _extract_dollar(line: str) -> Optional[float]:
    match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", line)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _parse_log_metrics(log_path: Path) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Log file not found: {log_path}") from exc


    nav = None
    gain = None
    annualized = None
    drawdown = None
    win_rate = None
    new_score_str = None
    new_score_pct = None
    worst_year_pct = None


    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "Score Result" in line:
            new_score_str = line
            match = re.search(r"\|\s*([0-9]+\.[0-9]+|[0-9]+)\s*\|$", line)
            if match:
                try:
                    new_score_pct = float(match.group(1))
                except Exception:
                    new_score_pct = None

        if "FINAL REALIZED CASH VALUE" in line or "FINAL ACCOUNT" in line:
            value = _extract_dollar(line)
            if value is not None:
                nav = value
        elif line.startswith("💵 **DAILY ACCOUNT VALUE"):
            value = _extract_dollar(line)
            if value is not None:
                nav = value

        if "TOTAL NET PROFIT (Start to Finish)" in line:
            value = _extract_dollar(line)
            if value is not None:
                gain = value
        elif "TOTAL NET REALIZED P&L (Cumulative)" in line and gain is None:
            value = _extract_dollar(line)
            if value is not None:
                gain = value

        if "Annualized Gain" in line:
            match = re.search(r"(-?\d+(?:\.\d+)?)%", line)
            if match:
                annualized = float(match.group(1))

        if "Worst Drawdown" in line:
            match = re.search(r"(-?\d+(?:\.\d+)?)%", line)
            if match:
                drawdown = float(match.group(1))

        if "Win Ratio" in line:
            match = re.search(r"(\d+(?:\.\d+)?)%", line)
            if match:
                win_rate = float(match.group(1))

        # Extract worst_year_pct from summary table row (actual format: ':  XX.XX%')
        if "Worst Year (Gain %)" in line:
            match = re.search(r":\s*([-+]?\d*\.\d+|\d+)%", line)
            if match:
                try:
                    worst_year_pct = float(match.group(1))
                except Exception:
                    worst_year_pct = None

    if nav is None:
        nav = _extract_float((FINAL_NAV_RE, DAILY_NAV_RE), text)
    if gain is None:
        gain = _extract_float((TOTAL_GAIN_RE, CUM_REALIZED_RE), text)
    if annualized is None:
        annualized = _extract_float((ANNUALIZED_GAIN_RE,), text)

    return nav, gain, annualized, drawdown, win_rate, new_score_pct, worst_year_pct


def _run_simulation_once(run_id: int) -> dict:
    if not SIMULATOR_PATH.exists():
        raise RuntimeError(f"Simulator not found at {SIMULATOR_PATH}")

    before_logs = _current_log_names()
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    result = subprocess.run(
        [sys.executable, str(SIMULATOR_PATH)],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.perf_counter() - start
    after_logs = _current_log_names()

    if result.returncode != 0:
        raise RuntimeError(
            f"simulate_yuda.py run {run_id} failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    log_path = _locate_log_file(result.stdout, before_logs, after_logs)
    nav, gain, annualized, drawdown, win_rate, new_score_pct, worst_year_pct = _parse_log_metrics(log_path)

    return {
        "run_id": run_id,
        "runtime_seconds": elapsed,
        "runtime": _format_runtime(elapsed),
        "nav": nav,
        "gain": gain,
        "annualized": annualized,
        "drawdown": drawdown,
        "win_rate": win_rate,
        "log_name": log_path.name,
        "score": new_score_pct,
        "new_score_pct": new_score_pct,
        "worst_year_pct": worst_year_pct,
        "param_value": None,
        "is_best": False,
    }


def _print_summary(rows: list[dict], param_name: str, emit: Callable[[str], None]) -> None:
    if not rows:
        emit(f"No simulation results to display for {param_name}.")
        return

    headers = [        
        "Run Id",
        param_name,
        "Run time",
        "Win Rate",        
        "$Gain",
        "Ann%",
        "Worst Year",
        "Drawdown",
        "Score",        
        "Log File",
    ]
    # Tripple ID is now assigned per triplet in main()

    table_rows = [
        [
            f"{row['tripple_id']}.{row['run_id']}",
            f"{row['param_display']} (best score)" if row.get("is_best") else str(row["param_display"]),
            row["runtime"],
            _format_pct(row.get("win_rate")),
            _format_money(row["gain"]),
            _format_pct(row["annualized"]),
            _format_pct(row.get("worst_year_pct")),
            _format_pct(row.get("drawdown")),
            _format_score(row.get("score")),            
            row["log_name"],
        ]
        for row in rows
    ]

    widths = [
        max(len(headers[i]), max(len(r[i]) for r in table_rows))
        for i in range(len(headers))
    ]

    def _print_line(columns: list[str]) -> None:
        emit(" | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))

    # Print total run-time since sim_wraper.py started
    global _sim_wraper_start_time
    if '_sim_wraper_start_time' in globals():
        elapsed = time.perf_counter() - _sim_wraper_start_time
        hh = int(elapsed // 3600)
        mm = int((elapsed % 3600) // 60)
        ss = int(elapsed % 60)
        emit(f"Sim Total Run-Time: {hh:02d}:{mm:02d}:{ss:02d}")
    else:
        emit("Sim Total Run-time: (unknown)")
    _print_line(headers)
    emit("-+-".join("-" * width for width in widths))
    for data, row in zip(table_rows, rows):
        _print_line(data) # Print Run1, Run2, Run3, Run4 in the table


def format_percent(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{int(round(value))}%"
    # Keep up to three decimal places while trimming unnecessary trailing zeros
    text = f"{value:.3f}"
    text = text.rstrip("0").rstrip(".")
    if text in {"", "-"}:
        text = "0"
    return f"{text}%"


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_float(value: float) -> str:
    formatted = f"{value:.4f}"
    formatted = formatted.rstrip("0").rstrip(".")
    if formatted == "":
        return "0"
    return formatted


def serialize_value(value: float, param_type: str) -> str:
    if param_type == "percent":
        return format_percent(value)
    if param_type == "currency":
        return format_currency(value)
    if param_type == "float":
        return format_float(value)
    return str(int(round(value)))


def compute_variants(base_value: float, param_type: str) -> tuple[float, float]:
    # Read wrapper_sweep_pct from rules.json (account_simulation section)
    try:
        with open(RULES_PATH, encoding="utf-8") as f:
            rules = json.load(f)
        sweep_str = rules.get("account_simulation", {}).get("wrapper_sweep_pct", "5%")
        if isinstance(sweep_str, str) and sweep_str.endswith("%"):
            wrapper_sweep_pct = float(sweep_str.rstrip("%")) / 100.0
        else:
            wrapper_sweep_pct = float(sweep_str)
    except Exception:
        wrapper_sweep_pct = 0.05

    if param_type == "int":
        plus = max(1, math.ceil(base_value * (1 + wrapper_sweep_pct)))    # +5% adjustment
        if math.isclose(plus, base_value, abs_tol=1e-9):
            plus = base_value + 1
        minus = max(1, math.floor(base_value * (1 - wrapper_sweep_pct)))  # -5% adjustment
        if math.isclose(minus, base_value, abs_tol=1e-9):
            minus = max(1, base_value - 1)
        return plus, minus

    elif param_type == "percent":
        precision = 3
        delta = 0.001
    elif param_type == "currency":
        precision = 2
        delta = 0.01
    else:  # float
        precision = 4
        delta = 0.0001

    plus = round(base_value * (1 + wrapper_sweep_pct), precision)  # +5% adjustment
    minus = round(base_value * (1 - wrapper_sweep_pct), precision) # -5% adjustment

    tol = 10 ** (-precision - 2)
    if math.isclose(plus, base_value, abs_tol=tol):
        plus = round(base_value + delta, precision)
    if math.isclose(minus, base_value, abs_tol=tol):
        minus = round(base_value - delta, precision)

    return plus, minus


def evaluate_and_mark(results: list[dict], default_value: float, param_type: str) -> float:
    best_row = None
    for row in results:
        score = row.get("new_score_pct")
        row["score"] = score
        if score is None:
            continue
        if best_row is None or (score is not None and score > best_row.get("score", float("-inf"))):
            best_row = row

    if best_row is None:
        best_value = default_value
    else:
        best_value = best_row["param_value"]

    for row in results:
        if param_type == "percent":
            row["is_best"] = math.isclose(row.get("param_value"), best_value, abs_tol=1e-9)
        else:
            row["is_best"] = row.get("param_value") == best_value
    return best_value


def main() -> None:
    original_rules_text = RULES_PATH.read_text(encoding="utf-8")
    original_rules = json.loads(original_rules_text)
    current_rules = deepcopy(original_rules)

    timestamp = datetime.now().strftime("%Y_%d_%m %H_%M")
    optimize_log_path = ROOT_DIR / "logs" / f"optimize {timestamp}.log"
    optimize_log_path.parent.mkdir(parents=True, exist_ok=True)
    with optimize_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"LOG FILE: {optimize_log_path.name}\n")
        log_file.write("=== ORIGINAL RULES ===\n")
        log_file.write(original_rules_text)
        log_file.write("\n")



    def log(message: str) -> None:
        if log.first:
            print(f"LOG FILE: {optimize_log_path.name}")
            log.first = False
        print(message)
        with optimize_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
    log.first = True

    def write_rules_file() -> None:
        serialized = json.dumps(current_rules, indent=4) + "\n"
        RULES_PATH.write_text(serialized, encoding="utf-8")

    def log_rules_snapshot(header: str) -> None:
        with optimize_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{header}\n")
            log_file.write(json.dumps(current_rules, indent=4))
            log_file.write("\n\n")

    try:
        param_specs = [
            # {"section": "account_simulation", "key": "max_puts_per_account", "label": "max_puts_per_account", "type": "int"},
            # {"section": "account_simulation", "key": "max_puts_per_stock", "label": "max_puts_per_stock", "type": "int"},
            {"section": "account_simulation", "key": "max_puts_per_day", "label": "max_puts_per_day", "type": "int"},
            {"section": "underlying_stock", "key": "min_5_day_rise_pct", "label": "min_5_day_rise_pct", "type": "percent"},
            {"section": "underlying_stock", "key": "min_above_avg_pct", "label": "min_above_avg_pct", "type": "percent"},
            {"section": "underlying_stock", "key": "max_above_avg_pct", "label": "max_above_avg_pct", "type": "percent"},
            {"section": "underlying_stock", "key": "min_avg_up_slope_pct", "label": "min_avg_up_slope_pct", "type": "percent"},
            {"section": "underlying_stock", "key": "min_stock_price", "label": "min_stock_price", "type": "currency"},
            {"section": "entry_put_position", "key": "min_days_for_expiration", "label": "min_days_for_expiration", "type": "int"},
            {"section": "entry_put_position", "key": "max_days_for_expiration", "label": "max_days_for_expiration", "type": "int"},
            {"section": "entry_put_position", "key": "min_put_bid_price", "label": "min_put_bid_price", "type": "float"},
            {"section": "entry_put_position", "key": "min_put_delta", "label": "min_put_delta", "type": "percent"},
            {"section": "entry_put_position", "key": "max_put_delta", "label": "max_put_delta", "type": "percent"},
            {"section": "entry_put_position", "key": "max_ask_above_bid_pct", "label": "max_ask_above_bid_pct", "type": "percent"},
            {"section": "entry_put_position", "key": "min_avg_above_strike", "label": "min_avg_above_strike", "type": "percent"},
            {"section": "entry_put_position", "key": "min_risk_reward_ratio", "label": "min_risk_reward_ratio", "type": "float"},
            {"section": "entry_put_position", "key": "min_annual_risk_reward_ratio", "label": "min_annual_risk_reward_ratio", "type": "float"},
            {"section": "entry_put_position", "key": "min_rev_annual_rr_ratio", "label": "min_rev_annual_rr_ratio", "type": "float"},
            {"section": "entry_put_position", "key": "min_expected_profit", "label": "min_expected_profit", "type": "percent"},            
            {"section": "exit_put_position", "key": "position_stop_loss_pct", "label": "position_stop_loss_pct", "type": "percent"},
            {"section": "exit_put_position", "key": "stock_max_below_avg", "label": "stock_max_below_avg", "type": "percent"},
            {"section": "exit_put_position", "key": "stock_max_below_entry", "label": "stock_max_below_entry", "type": "percent"},            
            {"section": "exit_put_position", "key": "min_gain_to_take_profit", "label": "min_gain_to_take_profit", "type": "percent"}
        ]

        tripple_id_counter = 1

        prev_best_result = None
        prev_best_param_value = None
        prev_best_param_display = None
        prev_best_log_name = None
        prev_best_runtime = None
        prev_best_runtime_seconds = None
        prev_best_win_rate = None
        prev_best_gain = None
        prev_best_annualized = None
        prev_best_drawdown = None
        prev_best_score = None

        for idx, spec in enumerate(param_specs):
            section = spec["section"]
            key = spec["key"]
            label = spec["label"]

            section = spec["section"]
            key = spec["key"]
            label = spec["label"]
            param_type = spec.get("type", "int")
            base_raw = current_rules.get(section, {}).get(key)
            if base_raw is None:
                raise KeyError(f"Missing '{label}' in rules.json")
            param_type = spec.get("type", "int")
            if param_type == "percent":
                base_value = float(str(base_raw).replace("%", ""))
            elif param_type == "currency":
                base_value = float(str(base_raw).replace("$", "").replace(",", ""))
            else:
                base_value = float(base_raw)

            plus_value, minus_value = compute_variants(base_value, param_type)


            candidates = []
            seen_values = set()
            for candidate in (base_value, plus_value, minus_value):
                if candidate not in seen_values:
                    candidates.append(candidate)
                    seen_values.add(candidate)

            # For possible Run4, store the extra +wrapper_sweep_pct and -wrapper_sweep_pct values
            try:
                sweep_str = current_rules.get("account_simulation", {}).get("wrapper_sweep_pct", "5%")
                if isinstance(sweep_str, str) and sweep_str.endswith("%"):
                    wrapper_sweep_pct_val = float(sweep_str.rstrip("%")) / 100.0
                else:
                    wrapper_sweep_pct_val = float(sweep_str)
            except Exception:
                wrapper_sweep_pct_val = 0.05

            extra_plus = plus_value * (1 + wrapper_sweep_pct_val) if param_type != "int" else math.ceil(plus_value * (1 + wrapper_sweep_pct_val))
            extra_minus = minus_value * (1 - wrapper_sweep_pct_val) if param_type != "int" else max(1, math.floor(minus_value * (1 - wrapper_sweep_pct_val)))

            log("")
            results: list[dict] = []

            run_results = []
            for run_id, value in enumerate(candidates, start=1):
                serialized_value = serialize_value(value, param_type)
                current_rules[section][key] = serialized_value
                write_rules_file()

                # FAKE RUN1 for triplets after the first
                if run_id == 1 and idx > 0 and prev_best_result is not None:
                    # Copy previous triplet's best result, update triplet/run IDs and param values
                    fake_result = prev_best_result.copy()
                    fake_result["run_id"] = 1
                    fake_result["tripple_id"] = tripple_id_counter
                    fake_result["param_value"] = value
                    fake_result["param_display"] = serialized_value
                    fake_result["is_best"] = True
                    results.append(fake_result)
                    run_results.append(fake_result)
                    # Print the starting message and the copied results
                    # Print Fake Run1 with Worst Year [%] as in real runs
                    worst_year_val = fake_result.get("worst_year_pct")
                    worst_year_str = f"{worst_year_val:.2f}%" if worst_year_val is not None else "N/A"
                    print(f"Run1 --> {label}={serialized_value}  ...    Ann: {_format_pct(fake_result.get('annualized'))}    Worst Year [%]: {worst_year_str}    Drawdown:{_format_pct(fake_result.get('drawdown'))}    Score:{_format_score(fake_result.get('score'))}")
                    continue

                # Print the starting message (no newline, flush immediately)
                print(f"Run{run_id} --> {label}={serialized_value}  ...", end="", flush=True)
                # Run simulation and capture result
                run_result = _run_simulation_once(run_id)
                run_result["param_value"] = value
                run_result["param_display"] = serialized_value
                run_result["tripple_id"] = tripple_id_counter
                results.append(run_result)
                run_results.append(run_result)
                # Prepare Ann% and Drawdown for printout
                ann_val = run_result.get("annualized")
                drawdown_val = run_result.get("drawdown")
                ann_str = _format_pct(ann_val) if ann_val is not None else "-"
                drawdown_str = _format_pct(drawdown_val) if drawdown_val is not None else "-"
                score_val = run_result.get("new_score_pct")
                score_str = f"{score_val:.4f}" if score_val is not None else "-"
                worst_year_val = run_result.get("worst_year_pct")
                worst_year_str = f"{worst_year_val:.2f}%" if worst_year_val is not None else "N/A"
                # Print results on the same line, including Worst Year [%]
                print(f"    Ann: {ann_str}    Worst Year [%]: {worst_year_str}    Drawdown:{drawdown_str}    Score:{score_str}")

            # Check if Run2 or Run3 is best, and if so, add a Run4 with additional +5% or -5%
            best_value = evaluate_and_mark(results, base_value, param_type)
            best_index = None
            for i, row in enumerate(results):
                if row.get("is_best"):
                    best_index = i
                    break

            run4_needed = False
            run4_value = None
            if best_index == 1:  # Run2 is best (index 1)
                run4_value = plus_value * 1.05 if param_type != "int" else math.ceil(plus_value * 1.05)
                run4_needed = True
            elif best_index == 2:  # Run3 is best (index 2)
                run4_value = minus_value * 0.95 if param_type != "int" else max(1, math.floor(minus_value * 0.95))
                run4_needed = True

            if run4_needed:
                serialized_value = serialize_value(run4_value, param_type)
                current_rules[section][key] = serialized_value
                write_rules_file()
                print(f"Run4 --> {label}={serialized_value}  ...", end="", flush=True)
                run_result = _run_simulation_once(4)
                run_result["param_value"] = run4_value
                run_result["param_display"] = serialized_value
                run_result["tripple_id"] = tripple_id_counter
                results.append(run_result)
                # Prepare Ann% and Drawdown for printout
                ann_val = run_result.get("annualized")
                drawdown_val = run_result.get("drawdown")
                ann_str = _format_pct(ann_val) if ann_val is not None else "-"
                drawdown_str = _format_pct(drawdown_val) if drawdown_val is not None else "-"
                # Use the best score from Run1, Run2, Run3
                best_score = max((r.get("score") for r in results[:3] if r.get("score") is not None), default=None)
                score_str = f"{best_score:.4f}" if best_score is not None else "-"
                print(f"    Ann: {ann_str}    Drawdown:{drawdown_str}    Score:{score_str}")

                # Re-evaluate best after Run4
                best_value = evaluate_and_mark(results, base_value, param_type)

            best_value = evaluate_and_mark(results, base_value, param_type)
            best_serialized = serialize_value(best_value, param_type)
            current_rules[section][key] = best_serialized
            write_rules_file()

            # Track score improvements
            global score_improvements_count
            if param_type == "percent":
                baseline_matched = math.isclose(best_value, base_value, abs_tol=1e-9)
            else:
                baseline_matched = best_value == base_value
            if not baseline_matched:
                score_improvements_count += 1

            # Always log rules.json after every triplet of runs (after each parameter sweep), but only once per sweep
            if idx == 0:
                log_rules_snapshot(f"--- rules.json after first sweep ({label}) ---")
            else:
                log_rules_snapshot(f"--- rules.json after sweep {idx+1} ({label}) ---")

            log("")
            _print_summary(results, label, log)
            log(f"Score improvements counted: {score_improvements_count}")
            tripple_id_counter += 1

            # Save the best result of this triplet for use in the next triplet's fake run
            prev_best_result = None
            for row in results:
                if row.get("is_best"):
                    prev_best_result = row.copy()
                    break

            if baseline_matched and idx < len(param_specs) - 1:
                log("")

    except Exception:
        RULES_PATH.write_text(original_rules_text, encoding="utf-8")
        raise

    with optimize_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("=== FINAL RULES ===\n")
        log_file.write(RULES_PATH.read_text(encoding="utf-8"))
        log_file.write("\n")
        log_file.write(f"LOG FILE: {optimize_log_path.name}\n")

        #if 'score_improvements_count' in locals():
        #    log_file.write(f"Score improvements counted: {score_improvements_count}\n")
        #else:
        #    log_file.write("Score improvements counted: 0\n")

        # Print to console as well
        #if 'score_improvements_count' in locals():
        #    print(f"Score improvements counted: {score_improvements_count}")
        #else:
        #    print("Score improvements counted: 0")
            
        log_file.write(f"LOG FILE: {optimize_log_path.name}\n")


if __name__ == "__main__":
    global _sim_wraper_start_time
    if not wrapper_sweep_pct_set or wrapper_sweep_pct_set == [0]:
        # Run once using the original wrapper_sweep_pct, do not overwrite rules.json
        _sim_wraper_start_time = time.perf_counter()
        main()
    else:
        for sweep_pct in wrapper_sweep_pct_set:
            _sim_wraper_start_time = time.perf_counter()
            # Overwrite wrapper_sweep_pct in rules.json
            try:
                with open(RULES_PATH, encoding="utf-8") as f:
                    rules = json.load(f)
                rules.setdefault("account_simulation", {})["wrapper_sweep_pct"] = f"{sweep_pct}%"
                with open(RULES_PATH, "w", encoding="utf-8") as f:
                    json.dump(rules, f, indent=4)
            except Exception as e:
                print(f"Failed to update wrapper_sweep_pct in rules.json: {e}")
            main()
