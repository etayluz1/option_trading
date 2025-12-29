# Version 12/28/2025 8:33 am    : 9999 and -9999 special flag handling for low_put_mode
# Version 12/28/2025 1:17 am    : fixed bug in Force LowPutsMode
# Version 12/28/2025 12:41 am   : Skip low_put_mode rules or Skip high_mode_rules
# Add random selctrion of wrapper_sweep_pct_set
# x.y.z is now doing y = 1, 2 to 15
import orjson
import math
import os
import random
import re
import subprocess
import sys
import time
import threading
import mmap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Optional

# Lock for log file identification and renaming to prevent race conditions
_log_file_lock = threading.Lock()

wrapper_sweep_pct_set = [0.3, 2, 3, 5, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 50]  # Percentages
wrap_group_size = 3  # Group size for optimization sweeps (1,2,3 or 4)
warp_set_size = 15

score_improvements_count = 0
baseline_result = None  # Store baseline simulation result for reuse

ROOT_DIR = Path(__file__).resolve().parent
SIMULATOR_PATH = ROOT_DIR / "sim.py"
LOGS_DIR = ROOT_DIR / "logs" 
RULES_PATH = ROOT_DIR / "rules.json"

# Match the console line emitted by sim.py with the generated log path.
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


def get_random_set(set_size: int) -> list[float]:
    """
    Generate a random set of sweep percentages.
    90% are integers from 1-50, 10% are fractions from [0.2, 0.4, 0.6, 0.8].
    Both selection and order are randomized.
    """
    fraction_count = max(1, round(set_size * 0.1))  # 10% fractions, at least 1
    integer_count = set_size - fraction_count       # 90% integers
    
    # Available pools
    integer_pool = list(range(1, 51))  # [1, 2, 3, ..., 50]
    fraction_pool = [0.2, 0.4, 0.6, 0.8]
    
    # Randomly select from each pool (without replacement for integers, with replacement for fractions if needed)
    selected_integers = random.sample(integer_pool, min(integer_count, len(integer_pool)))
    selected_fractions = random.choices(fraction_pool, k=fraction_count)
    
    # Combine and shuffle
    result = selected_integers + selected_fractions
    random.shuffle(result)
    
    return result

# Generate random sweep percentages (must be after function definition)
wrapper_sweep_pct_set = get_random_set(warp_set_size)


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
    if value is None:
        return "N/A"
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


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
    match = re.search(r"\$\s*(-?[\d,]+(?:\.\d+)?)", line)
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
            # Match the value in the '| Score Result ... |   value   |' line, including negative values
            match = re.search(r"\|\s*Score Result.*?\|\s*(-?[0-9]+\.[0-9]+|-?[0-9]+)\s*\|", line)
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


def _run_simulation_once(rule_id: int, wrap_id: int, try_id: int, trial_timestamp: str, temp_rules_path: Path = None) -> dict:
    """Run a single simulation with x.y.z naming (Rule_ID.Wrap_ID.Try_ID)"""
    if not SIMULATOR_PATH.exists():
        raise RuntimeError(f"Simulator not found at {SIMULATOR_PATH}")

    before_logs = _current_log_names()
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Pass the complete log filename to sim.py
    log_filename = f"{trial_timestamp} {rule_id}-{wrap_id}-{try_id}.log"
    env["SIM_WRAPPER_LOG_FILENAME"] = log_filename
    # Pass the temporary rules file path to sim.py (use absolute path)
    if temp_rules_path:
        abs_path = temp_rules_path.resolve()
        env["SIM_WRAPPER_RULES_PATH"] = str(abs_path)
    # Pass stock_history as JSON string if available
    if hasattr(main, "stock_history_json_str"):
        env["SIM_WRAPPER_STOCK_HISTORY_JSON"] = main.stock_history_json_str

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
            f"sim.py trial {rule_id}.{wrap_id}.{try_id} failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # Use lock to prevent race conditions when multiple runs try to identify log files
    with _log_file_lock:
        log_path = _locate_log_file(result.stdout, before_logs, after_logs)
    
    nav, gain, annualized, drawdown, win_rate, new_score_pct, worst_year_pct = _parse_log_metrics(log_path)

    return {
        "rule_id": rule_id,
        "wrap_id": wrap_id,
        "try_id": try_id,
        "trial_id": f"{rule_id}.{wrap_id}.{try_id}",
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
        "Trial ID",
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

    table_rows = [
        [
            f"{row['trial_id']}",
            f"{row['param_display']} (best)" if row.get("is_best") else str(row["param_display"]),
            row["runtime"],
            _format_pct(row.get("win_rate")),
            _format_money(row["gain"]),
            _format_pct(row.get("annualized")),
            _format_pct(row.get("worst_year_pct")),
            _format_pct(row.get("drawdown")),
            _format_score(row.get("score")),            
            f"{row.get('log_name', 'N/A')}  (reused)" if row.get("is_reused") else row.get("log_name", "N/A"),
        ]
        for row in rows
    ]

    widths = [
        max(len(headers[i]), max(len(r[i]) for r in table_rows))
        for i in range(len(headers))
    ]

    def _print_line(columns: list[str]) -> None:
        emit(" | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))

    # Print total run-time since wrap.py started    
    if '_wrap_start_time' in globals():
        elapsed = time.perf_counter() - _wrap_start_time
        hh = int(elapsed // 3600)
        mm = int((elapsed % 3600) // 60)
        ss = int(elapsed % 60)
        emit(f"Sim Total Run-Time: {hh:02d}:{mm:02d}:{ss:02d}")
    else:
        emit("Sim Total Run-time: (unknown)")
    _print_line(headers)
    emit("-+-".join("-" * width for width in widths))
    for data, row in zip(table_rows, rows):
        _print_line(data)


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


def compute_variants_for_wrap_id(base_value: float, param_type: str, sweep_pct: float) -> tuple[float, float]:
    """Compute plus/minus variants for a specific sweep percentage"""
    wrapper_sweep_pct = sweep_pct / 100.0  # Convert percentage to decimal

    if param_type == "int":
        plus = max(1, math.ceil(base_value * (1 + wrapper_sweep_pct)))
        if math.isclose(plus, base_value, abs_tol=1e-9):
            plus = base_value + 1
        minus = max(1, math.floor(base_value * (1 - wrapper_sweep_pct)))
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

    plus = round(base_value * (1 + wrapper_sweep_pct), precision)
    minus = round(base_value * (1 - wrapper_sweep_pct), precision)

    tol = 10 ** (-precision - 2)
    if math.isclose(plus, base_value, abs_tol=tol):
        plus = round(base_value + delta, precision)
    if math.isclose(minus, base_value, abs_tol=tol):
        minus = round(base_value - delta, precision)

    return plus, minus


def main(wrapper_sweep_pct_group: list[float], global_wrap_idx_start: int = 1) -> None:
    # Load stock_history.json once and serialize as JSON string for env transfer
    stock_history_path = ROOT_DIR / "stock_history.json"
    with stock_history_path.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            stock_history_data = mmapped_file[:]
    stock_history_dict = orjson.loads(stock_history_data)
    stock_history_json_str = orjson.dumps(stock_history_dict).decode("utf-8")
    main.stock_history_json_str = stock_history_json_str

    # ...existing code...
    original_rules_text = RULES_PATH.read_text(encoding="utf-8")
    original_rules = orjson.loads(original_rules_text)
    current_rules = deepcopy(original_rules)

    timestamp = datetime.now().strftime("%Y_%m_%d %H_%M")
    optimize_log_path = ROOT_DIR / "logs" / f"optimize {timestamp}.log"
    optimize_log_path.parent.mkdir(parents=True, exist_ok=True)
    with optimize_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"LOG FILE: {optimize_log_path.name}\n")
        log_file.write("=== ORIGINAL RULES ===\n")
        log_file.write(original_rules_text)
        log_file.write("\n")
        log_file.write(f"Wrapper sweep percentages: {wrapper_sweep_pct_group}\n\n")

    def log(message: str) -> None:
        if log.first:
            print(f"LOG FILE: {optimize_log_path.name}")
            log.first = False
        print(message)
        with optimize_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
    log.first = True

    def write_rules_file(wrapper_sweep_pct_value=None):
        # Always update wrapper_sweep_pct in rules before saving
        if wrapper_sweep_pct_value is not None:
            current_rules["account_simulation"]["wrapper_sweep_pct"] = f"{wrapper_sweep_pct_value}%"
        serialized = orjson.dumps(current_rules, option=orjson.OPT_INDENT_2).decode("utf-8") + "\n"
        RULES_PATH.write_text(serialized, encoding="utf-8")

    def log_rules_snapshot(header: str) -> None:
        with optimize_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{header}\n")
            log_file.write(orjson.dumps(current_rules, option=orjson.OPT_INDENT_2).decode("utf-8"))
            log_file.write("\n\n")

    try:
        param_specs = [
            {"section": "account_simulation", "key": "max_puts_per_account", "label": "max_puts_per_account", "type": "int"},
            {"section": "account_simulation", "key": "max_puts_per_stock", "label": "max_puts_per_stock", "type": "int"},
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
            {"section": "exit_put_position", "key": "min_gain_to_take_profit", "label": "min_gain_to_take_profit", "type": "percent"},
            {"section": "low_put_mode", "key": "low_min_puts_to_set_low_mode", "label": "low_min_puts_to_set_low_mode", "type": "int"},
            {"section": "low_put_mode", "key": "low_max_puts_to_set_high_mode", "label": "low_max_puts_to_set_high_mode", "type": "int"},
            {"section": "low_put_mode", "key": "low_max_puts_per_account", "label": "low_max_puts_per_account", "type": "int"},
            {"section": "low_put_mode", "key": "low_max_puts_per_stock", "label": "low_max_puts_per_stock", "type": "int"},
            {"section": "low_put_mode", "key": "low_max_puts_per_day", "label": "low_max_puts_per_day", "type": "int"},
            {"section": "low_put_mode", "key": "low_min_5_day_rise_pct", "label": "low_min_5_day_rise_pct", "type": "percent"},
            {"section": "low_put_mode", "key": "low_min_above_avg_pct", "label": "low_min_above_avg_pct", "type": "percent"},
            {"section": "low_put_mode", "key": "low_max_above_avg_pct", "label": "low_max_above_avg_pct", "type": "percent"},
            {"section": "low_put_mode", "key": "low_min_avg_up_slope_pct", "label": "low_min_avg_up_slope_pct", "type": "percent"},
            {"section": "low_put_mode", "key": "low_min_stock_price", "label": "low_min_stock_price", "type": "currency"},
            {"section": "low_put_mode", "key": "low_max_days_for_expiration", "label": "low_max_days_for_expiration", "type": "int"},
            {"section": "low_put_mode", "key": "low_min_put_bid_price", "label": "low_min_put_bid_price", "type": "float"},
            {"section": "low_put_mode", "key": "low_min_put_delta", "label": "low_min_put_delta", "type": "percent"},
            {"section": "low_put_mode", "key": "low_max_put_delta", "label": "low_max_put_delta", "type": "percent"},
            {"section": "low_put_mode", "key": "low_max_ask_above_bid_pct", "label": "low_max_ask_above_bid_pct", "type": "percent"},
            {"section": "low_put_mode", "key": "low_min_risk_reward_ratio", "label": "low_min_risk_reward_ratio", "type": "float"},
            {"section": "low_put_mode", "key": "low_min_annual_risk_reward_ratio", "label": "low_min_annual_risk_reward_ratio", "type": "float"},
            {"section": "low_put_mode", "key": "low_min_rev_annual_rr_ratio", "label": "low_min_rev_annual_rr_ratio", "type": "float"},
            {"section": "low_put_mode", "key": "low_min_expected_profit", "label": "low_min_expected_profit", "type": "percent"},
            {"section": "low_put_mode", "key": "low_position_stop_loss_pct", "label": "low_position_stop_loss_pct", "type": "percent"},
            {"section": "low_put_mode", "key": "low_stock_max_below_avg", "label": "low_stock_max_below_avg", "type": "percent"},
            {"section": "low_put_mode", "key": "low_stock_max_below_entry", "label": "low_stock_max_below_entry", "type": "percent"},
            {"section": "low_put_mode", "key": "low_min_gain_to_take_profit", "label": "low_min_gain_to_take_profit", "type": "percent"}
        ]

        # --- Determine skip modes based on rules.json values ---
        # skip_low_mode: if low_min_puts_to_set_low_mode == "9999", low mode is disabled, skip all low_put_mode rules
        # skip_high_mode: if low_max_puts_to_set_high_mode == "9999", always in low mode, skip regular rules with low equivalents
        low_min_puts_val = str(current_rules.get("low_put_mode", {}).get("low_min_puts_to_set_low_mode", "")).strip()
        low_max_puts_val = str(current_rules.get("low_put_mode", {}).get("low_max_puts_to_set_high_mode", "")).strip()
        skip_low_mode = (low_min_puts_val == "9999")
        skip_high_mode = (low_max_puts_val == "9999")
        
        # Build set of low_put_mode keys (without "low_" prefix) for skip_high_mode matching
        low_mode_keys_without_prefix = set()
        for spec in param_specs:
            if spec["section"] == "low_put_mode" and spec["key"].startswith("low_"):
                # Remove "low_" prefix to get the equivalent regular key name
                low_mode_keys_without_prefix.add(spec["key"][4:])  # e.g., "low_max_puts_per_account" -> "max_puts_per_account"
        
        if skip_low_mode:
            log(f"⚡ skip_low_mode=True (low_min_puts_to_set_low_mode=9999): Will skip all low_put_mode rules")
        if skip_high_mode:
            log(f"⚡ skip_high_mode=True (low_max_puts_to_set_high_mode=9999): Will skip regular rules with low equivalents")

        global score_improvements_count, baseline_result
        prev_best_result = baseline_result.copy() if baseline_result else None
        baseline_score = baseline_result.get("new_score_pct") if baseline_result else None
        
        # For each Rule_ID (parameter)
        for rule_id, spec in enumerate(param_specs, start=1):
            section = spec["section"]
            key = spec["key"]
            label = spec["label"]
            param_type = spec.get("type", "int")
            
            # Get base value from current rules
            # CRITICAL: Read base value from ORIGINAL_RULES, not current_rules
            # current_rules gets modified as we optimize each rule, so later rules would inherit
            # the optimized values from earlier rules, causing values to "leak" between rules
            base_raw = original_rules.get(section, {}).get(key)
            if base_raw is None:
                raise KeyError(f"Missing '{label}' in rules.json")
            
            if param_type == "percent":
                base_value = float(str(base_raw).replace("%", ""))
            elif param_type == "currency":
                base_value = float(str(base_raw).replace("$", "").replace(",", ""))
            else:
                base_value = float(base_raw)
            
            # Note: prev_best_result is only used to reuse simulation results for x.1.1 trials,
            # NOT to override the base_value for this parameter
            
            log("")
            log(f"=" * 80)
            log(f"Rule {rule_id}/{len(param_specs)}: {label}")
            log(f"=" * 80)
            
            # Prepare 9-trial set (or less if wrapper_sweep_pct_group has < 3 values)
            # Format: [x.1.1, x.1.2, x.1.3, x.2.1, x.2.2, x.2.3, x.3.1, x.3.2, x.3.3]
            # All trials run concurrently
            
            trial_tasks = []
            trial_timestamp = datetime.now().strftime("%y-%m-%d %H-%M")
            assigned_values = set()
            
            # Rotate wrap_id offset based on rule progress to vary testing pattern
            wrap_id_offset = (rule_id - 1) % wrap_group_size
            
            for local_idx, sweep_pct in enumerate(wrapper_sweep_pct_group, start=1):
                # Find the 1-based index of sweep_pct in the full wrapper_sweep_pct_set
                try:
                    sweep_y = wrapper_sweep_pct_set.index(sweep_pct) + 1
                except ValueError:
                    sweep_y = local_idx  # fallback to local_idx if not found
                # Compute variants for this sweep_pct
                plus_value, minus_value = compute_variants_for_wrap_id(base_value, param_type, sweep_pct)

                # Prepare all three trial values for this sweep_pct
                for try_id, value in zip([1,2,3], [base_value, plus_value, minus_value]):
                    # Skip try_id=1 (-->0%) for Rule 2+
                    # Only include the FIRST try_id=1 as a reused fake run
                    if rule_id > 1 and try_id == 1:
                        if local_idx == 1 and prev_best_result:
                            # First wrap: will inject as reused result
                            serialized_value = serialize_value(value, param_type)
                            print(f"{rule_id}.{sweep_y}.{try_id} --> {label}={serialized_value} (reused from Rule {prev_best_result['rule_id']})", flush=True)
                        # Skip adding to trial_tasks (all other try_id=1 or first one will be injected)
                        continue
                    # Use string representation for set to handle float/int/str uniformly
                    value_key = str(value)
                    if value_key in assigned_values:
                        continue  # Skip if this value was already assigned in this trial set
                    assigned_values.add(value_key)
                    # Determine if this rule should be skipped
                    is_skipped = False
                    if skip_low_mode and section == "low_put_mode":
                        is_skipped = True  # Skip all low_put_mode rules when low mode is disabled
                    elif skip_high_mode and section != "low_put_mode" and key in low_mode_keys_without_prefix:
                        is_skipped = True  # Skip regular rules that have low equivalents when always in low mode
                    
                    # Skip the two mode flag rules if they have special flag values (9999 or -9999)
                    if key in ["low_min_puts_to_set_low_mode", "low_max_puts_to_set_high_mode"]:
                        flag_val = str(current_rules.get("low_put_mode", {}).get(key, "")).strip()
                        if flag_val in ["9999", "-9999"]:
                            is_skipped = True  # These are special flags, not values to optimize
                    
                    trial_tasks.append({
                        "rule_id": rule_id,  # x: 1 to 46
                        "wrap_id": sweep_y,  # y: 1 to 15 (index in wrapper_sweep_pct_set)
                        "try_id": try_id,    # z: 1 to 3
                        "value": value,
                        "sweep_pct": sweep_pct,
                        "label": label,
                        "section": section,
                        "key": key,
                        "param_type": param_type,
                        "is_reused": False,
                        "is_skipped": is_skipped
                    })
            
            # Special case: If rule_id=1 and all trials are skipped, we must run one baseline with wrap=0
            # to establish a baseline for future rules
            if rule_id == 1:
                all_skipped = all(task.get("is_skipped", False) for task in trial_tasks)
                if all_skipped and len(trial_tasks) > 0:
                    # Force run the first trial (1.1.1) with wrap=0 as baseline
                    log("⚠️  All Rule 1 trials are skipped, but we need a baseline. Forcing 1.1.1 with wrap=0.")
                    trial_tasks[0]["is_skipped"] = False
            
            # Execute 9 trials concurrently
            results = []
            
            def run_single_trial(task):
                rule_id = task["rule_id"]  # x
                wrap_id = task["wrap_id"]  # y (should be 1-15, index in wrapper_sweep_pct_set)
                try_id = task["try_id"]    # z

                value = task["value"]
                sweep_pct = task["sweep_pct"]
                label = task["label"]
                section = task["section"]
                key = task["key"]
                param_type = task["param_type"]
                is_reused = task["is_reused"]
                is_skipped = task.get("is_skipped", False)

                serialized_value = serialize_value(value, param_type)
                # y is always the 1-based index in wrapper_sweep_pct_set
                trial_id = f"{rule_id}.{wrap_id}.{try_id}"

                # If this rule is skipped (disabled), return early with dummy result
                if is_skipped:
                    skip_reason = "low mode disabled" if section == "low_put_mode" else "has low equivalent"
                    print(f"{trial_id} --> {label}={serialized_value} (SKIPPED: {skip_reason})", flush=True)
                    return {
                        "rule_id": rule_id,
                        "wrap_id": wrap_id,
                        "try_id": try_id,
                        "trial_id": trial_id,
                        "param_value": value,
                        "param_display": serialized_value,
                        "is_skipped": True,
                        "score": None,
                        "gain": None,
                        "runtime": "SKIPPED"
                    }

                # If this is a reused trial, skip simulation and use previous rule's result
                if is_reused and prev_best_result:
                    result = prev_best_result.copy()
                    # Update fields to reflect current rule, but keep all metrics from previous rule
                    result["rule_id"] = rule_id
                    result["wrap_id"] = wrap_id
                    result["try_id"] = try_id
                    result["trial_id"] = trial_id
                    result["param_value"] = value
                    result["param_display"] = serialized_value
                    result["is_reused"] = True
                    # Ensure all required fields for table display are present
                    if "runtime" not in result:
                        result["runtime"] = result.get("runtime", "N/A")
                    if "gain" not in result:
                        result["gain"] = result.get("gain")
                    if "score" not in result:
                        result["score"] = result.get("score")
                    print(f"{trial_id} --> {label}={serialized_value} (reused from Rule {prev_best_result['rule_id']})", flush=True)
                    return result

                # Create temporary rules for this trial
                temp_rules = deepcopy(current_rules)
                temp_rules[section][key] = serialized_value
                # Update wrapper_sweep_pct to reflect the actual sweep percentage being used
                temp_rules["account_simulation"]["wrapper_sweep_pct"] = f"{sweep_pct}%"
                # Include wrapper_sweep_pct in the log file name
                sweep_pct_str = str(sweep_pct).replace('.', '_')
                temp_rules_path = ROOT_DIR / f"rules_temp_{rule_id}_{wrap_id}_{try_id}.json"
                try:
                    # Write temporary rules file
                    with temp_rules_path.open("wb") as f:
                        f.write(orjson.dumps(temp_rules, option=orjson.OPT_INDENT_2))
                    # Small staggered delay to ensure unique log file timestamps
                    # and prevent concurrent file access issues
                    time.sleep((wrap_id * 3 + try_id) * 0.5)  # Increased from 0.05 to 0.5 seconds
                    # Display sweep indicator (show -->0% for baseline, +% for plus, -% for minus)
                    if try_id == 1:
                        sweep_display = f"{sweep_pct}% -->0%"
                    elif try_id == 2:
                        sweep_display = f"+{sweep_pct}%"
                    else:  # try_id == 3
                        sweep_display = f"-{sweep_pct}%"
                    print(f"{trial_id} --> {label}={serialized_value} (sweep={sweep_display})", flush=True)
                    # Run simulation with temp rules file (passed via environment variable)
                    result = _run_simulation_once(
                        rule_id, wrap_id, try_id, trial_timestamp, temp_rules_path
                    )
                    result["param_value"] = value
                    result["param_display"] = serialized_value
                    # Check if simulation produced valid results
                    if result.get('score') is None:
                        print(f"  WARNING: Simulation completed but produced no valid score!")
                    # ...existing code...
                    return result
                finally:
                    # Clean up temp rules file
                    if temp_rules_path.exists():
                        temp_rules_path.unlink()
            
            # Run all trials in parallel (max 9 concurrent workers for 3 sweep percentages)
            
            print(f"Starting {len(trial_tasks)}-trial set for Rule {rule_id}: {label}")
            
            with ThreadPoolExecutor(max_workers=min(wrap_group_size * 3, len(trial_tasks))) as executor:
                future_to_task = {executor.submit(run_single_trial, task): task for task in trial_tasks}
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        error_msg = str(e)
                        # Windows access violation error (0xC0000005 = 3221225477)
                        if "3221225477" in error_msg or "exit code -1073741819" in error_msg:
                            print(f"\n⚠️  Trial {task['rule_id']}.{task['wrap_id']}.{task['try_id']} crashed (access violation). This may be due to memory corruption or invalid parameter combination.")
                        else:
                            print(f"\n❌ Error in trial {task['rule_id']}.{task['wrap_id']}.{task['try_id']}: {e}")
                        # Continue with other trials instead of crashing
                        continue
            
            # If we skipped all try_id=1 for Rule 2+, inject the reused result for the first wrap
            if rule_id > 1 and prev_best_result and len(wrapper_sweep_pct_group) > 0:
                # The first sweep_pct in the group
                first_sweep_pct = wrapper_sweep_pct_group[0]
                try:
                    first_sweep_y = wrapper_sweep_pct_set.index(first_sweep_pct) + 1
                except ValueError:
                    first_sweep_y = 1
                base_raw = original_rules.get(section, {}).get(key)
                if param_type == "percent":
                    base_value = float(str(base_raw).replace("%", ""))
                elif param_type == "currency":
                    base_value = float(str(base_raw).replace("$", "").replace(",", ""))
                else:
                    base_value = float(base_raw)
                serialized_value = serialize_value(base_value, param_type)
                reused_result = prev_best_result.copy()
                reused_result["rule_id"] = rule_id
                reused_result["wrap_id"] = first_sweep_y
                reused_result["try_id"] = 1
                reused_result["trial_id"] = f"{rule_id}.{first_sweep_y}.1"
                reused_result["param_value"] = base_value
                reused_result["param_display"] = serialized_value
                reused_result["is_reused"] = True
                results.append(reused_result)
            
            
            # Sort results: prioritize try_id=1 first (reused baseline), then by trial_id order
            results.sort(key=lambda x: (x["rule_id"], x["try_id"] != 1, x["wrap_id"], x["try_id"]))
            
            # Find best result
            best_result = None
            best_score = float("-inf")
            for result in results:
                score = result.get("new_score_pct")
                if score is not None and score > best_score:
                    best_score = score
                    best_result = result
            
            # If there's a reused baseline and all scores are the same, mark the reused as best
            if rule_id > 1:
                for result in results:
                    if result.get("is_reused"):
                        reused_score = result.get("new_score_pct")
                        if reused_score == best_score:
                            best_result = result
                        break
            
            # Mark only the best result
            for result in results:
                result["is_best"] = (result is best_result)
            
            # Update current rules with best value
            if best_result:
                best_value = best_result["param_value"]
                best_serialized = serialize_value(best_value, param_type)
                current_rules[section][key] = best_serialized
                # Save the current wrapper_sweep_pct value in rules.json
                write_rules_file(wrapper_sweep_pct_value=best_result.get("sweep_pct"))
                # Check if this is an improvement over the baseline score (not just parameter change)
                best_score = best_result.get("new_score_pct")
                if rule_id == 1:
                    # Store the first rule's best score as the baseline for all future comparisons
                    baseline_result = best_result.copy()
                    baseline_score = best_score
                    # First rule doesn't count as an improvement (it's the baseline)
                elif baseline_score is not None and best_score is not None and best_score > baseline_score and not best_result.get("is_reused"):
                    # Only increment if this rule's best score beats the baseline score AND is not a reused (fake) baseline
                    score_improvements_count += 1
                # Save best result for next rule
                prev_best_result = best_result.copy()
            
            # Log results
            log("")
            _print_summary(results, label, log)
            log(f"Best value: {best_result['param_display'] if best_result else 'N/A'}")
            log(f"Score improvements so far: {score_improvements_count}")
            
            # Log rules snapshot after each rule
            log_rules_snapshot(f"--- rules.json after Rule {rule_id} ({label}) ---")
    
    except Exception:
        RULES_PATH.write_text(original_rules_text, encoding="utf-8")
        raise
    
    with optimize_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("=== FINAL RULES ===\n")
        log_file.write(RULES_PATH.read_text(encoding="utf-8"))
        log_file.write("\n")
        log_file.write(f"LOG FILE: {optimize_log_path.name}\n")


if __name__ == "__main__":
    global _wrap_start_time
    _wrap_start_time = time.perf_counter()
    # ...existing code...
    
    # Break wrapper_sweep_pct_set into groups of wrap_group_size
    if len(wrapper_sweep_pct_set) > wrap_group_size:
        groups = [wrapper_sweep_pct_set[i:i+wrap_group_size] for i in range(0, len(wrapper_sweep_pct_set), wrap_group_size)]
        print(f"wrapper_sweep_pct_set has {len(wrapper_sweep_pct_set)} values, breaking into {len(groups)} groups of up to {wrap_group_size}")
    else:
        groups = [wrapper_sweep_pct_set]
    
    global_wrap_idx = 1  # Track global wrap index across all groups
    for group_idx, group in enumerate(groups, start=1):
        print()
        print(f"Starting optimization group {group_idx}/{len(groups)}: sweep percentages = {group}")
        print()
        main(group, global_wrap_idx_start=global_wrap_idx)
        global_wrap_idx += len(group)  # Increment for next group
    
    elapsed = time.perf_counter() - _wrap_start_time
    hh = int(elapsed // 3600)
    mm = int((elapsed % 3600) // 60)
    ss = int(elapsed % 60)
    print(f"*  *  *  *  *   *  *  *  *  *  *  D  O  N  E  *  *  *  * {hh:02d}:{mm:02d}:{ss:02d}  *  *   *  *  *  *  *  *  *  *  *  *")
    print()
