import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parent
SIMULATOR_PATH = ROOT_DIR / "simulate_new.py"
LOGS_DIR = ROOT_DIR / "logs"
RULES_PATH = ROOT_DIR / "rules.json"

# Match the console line emitted by simulate_new.py with the generated log path.
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
    return "N/A" if value is None else f"{value:.2f}%"


def _format_score(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


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


def _parse_log_metrics(log_path: Path) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Log file not found: {log_path}") from exc

    nav = None
    gain = None
    annualized = None
    drawdown = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

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

    if nav is None:
        nav = _extract_float((FINAL_NAV_RE, DAILY_NAV_RE), text)
    if gain is None:
        gain = _extract_float((TOTAL_GAIN_RE, CUM_REALIZED_RE), text)
    if annualized is None:
        annualized = _extract_float((ANNUALIZED_GAIN_RE,), text)

    return nav, gain, annualized, drawdown


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
            f"simulate_new.py run {run_id} failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    log_path = _locate_log_file(result.stdout, before_logs, after_logs)
    nav, gain, annualized, drawdown = _parse_log_metrics(log_path)

    return {
        "run_id": run_id,
        "runtime_seconds": elapsed,
        "runtime": _format_runtime(elapsed),
        "nav": nav,
        "gain": gain,
        "annualized": annualized,
        "drawdown": drawdown,
        "log_name": log_path.name,
        "score": None,
        "param_value": None,
        "is_best": False,
    }


def _print_summary(rows: list[dict], param_name: str) -> None:
    if not rows:
        print(f"No simulation results to display for {param_name}.")
        return

    headers = [
        "Run Id",
        param_name,
        "Run time",
        "$NAV",
        "$Gain",
        "Ann%",
        "Drawdown",
        "Score",
        "Log File",
    ]
    table_rows = [
        [
            str(row["run_id"]),
            (
                f"{row['param_display']} (best score)"
                if row.get("is_best")
                else str(row["param_display"])
            ),
            row["runtime"],
            _format_money(row["nav"]),
            _format_money(row["gain"]),
            _format_pct(row["annualized"]),
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
        print(" | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))

    print("Simulation summary:")
    _print_line(headers)
    print("-+-".join("-" * width for width in widths))
    for data in table_rows:
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


def compute_variants(base_value: float, param_type: str) -> tuple[float, float]:
    if param_type == "int":
        plus = max(1, math.ceil(base_value * 1.05))
        if math.isclose(plus, base_value, abs_tol=1e-9):
            plus = base_value + 1
        minus = max(1, math.floor(base_value * 0.95))
        if math.isclose(minus, base_value, abs_tol=1e-9):
            minus = max(1, base_value - 1)
        return plus, minus

    if param_type == "percent":
        precision = 3
        delta = 0.001
    elif param_type == "currency":
        precision = 2
        delta = 0.01
    else:  # float
        precision = 4
        delta = 0.0001

    plus = round(base_value * 1.05, precision)
    minus = round(base_value * 0.95, precision)

    tol = 10 ** (-precision - 2)
    if math.isclose(plus, base_value, abs_tol=tol):
        plus = round(base_value + delta, precision)
    if math.isclose(minus, base_value, abs_tol=tol):
        minus = round(base_value - delta, precision)

    return plus, minus


def evaluate_and_mark(results: list[dict], default_value: float, param_type: str) -> float:
    best_row = None
    for row in results:
        ann = row.get("annualized")
        drawdown = row.get("drawdown")
        if (
            ann is None
            or drawdown is None
            or drawdown >= 0
            or abs(drawdown) < 1e-9
        ):
            row["score"] = None
            continue
        score = ann / abs(drawdown)
        row["score"] = score
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

    def write_rules_file() -> None:
        RULES_PATH.write_text(json.dumps(current_rules, indent=4) + "\n", encoding="utf-8")

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
            {"section": "entry_put_position", "key": "min_expected_profit", "label": "min_expected_profit", "type": "percent"},
            {"section": "exit_put_position", "key": "stock_max_below_avg", "label": "stock_max_below_avg", "type": "percent"},
            {"section": "exit_put_position", "key": "stock_max_below_entry", "label": "stock_max_below_entry", "type": "percent"},
        ]

        for idx, spec in enumerate(param_specs):
            section = spec["section"]
            key = spec["key"]
            label = spec["label"]

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

            results: list[dict] = []
            for run_id, value in enumerate(candidates, start=1):
                serialized_value = serialize_value(value, param_type)
                current_rules[section][key] = serialized_value
                write_rules_file()
                print(f"Starting run {run_id} with {label}={serialized_value}...")
                run_result = _run_simulation_once(run_id)
                run_result["param_value"] = value
                run_result["param_display"] = serialized_value
                results.append(run_result)

            best_value = evaluate_and_mark(results, base_value, param_type)
            best_serialized = serialize_value(best_value, param_type)
            current_rules[section][key] = best_serialized
            write_rules_file()
            print(f"Applied best {label}={best_serialized} based on score.")

            print()
            _print_summary(results, label)

            if param_type == "percent":
                baseline_matched = math.isclose(best_value, base_value, abs_tol=1e-9)
            else:
                baseline_matched = best_value == base_value

            if baseline_matched and idx < len(param_specs) - 1:
                print()
                print(f"Best configuration matched baseline; optimizing {param_specs[idx + 1]['label']}...")

    except Exception:
        RULES_PATH.write_text(original_rules_text, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
