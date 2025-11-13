import sys
import os
from datetime import datetime
import orjson
from simulation_engine import run_simulation_in_memory

# --- Logger Class ---
class Logger:
    """Redirects print statements to both the console and a log file."""
    def __init__(self, filepath, minimal_mode=False):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')
        self.minimal_mode = minimal_mode
        self.line_count = 0
        self.startup_shown = False
        # In minimal mode, suppress console output initially but will enable after line 73760
        self.suppress_output = minimal_mode

    def write(self, message):
        # Always write to logfile with proper UTF-8 encoding
        self.logfile.write(message)
        
        # Count lines (each message could contain multiple lines)
        line_increments = message.count('\n')
        if line_increments > 0:
            self.line_count += line_increments
        
        # Show startup message once in minimal mode
        if self.minimal_mode and not self.startup_shown:
            self.terminal.write("The simulation is starting now\n")
            self.terminal.flush()
            self.startup_shown = True
        
        # In minimal mode, only show monthly summaries and content after line 73760
        if self.minimal_mode:
            # Check if this is a monthly summary (contains "Account Value:" and "RunTime:")
            is_monthly_summary = "Account Value:" in message and "RunTime:" in message
            # Enable console output after line 73760
            is_late_content = self.line_count >= 73760
            
            if not (is_monthly_summary or is_late_content):
                return  # Skip console output for this message
        
        # Console output (conditional based on suppress_output flag)
        if not self.suppress_output:
            try:
                self.terminal.write(message)
            except UnicodeEncodeError:
                # Remove emojis for Windows console compatibility
                import re
                safe_message = re.sub(r'[\u2600-\u26FF\u2700-\u27BF\U0001F000-\U0001F9FF]', '', message)
                self.terminal.write(safe_message)

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
        try:
            self.terminal.write(message + '\n')
            self.terminal.flush()
        except UnicodeEncodeError:
            # Remove emojis for Windows console compatibility
            import re
            safe_message = re.sub(r'[\u2600-\u26FF\u2700-\u27BF\U0001F000-\U0001F9FF]', '', message)
            self.terminal.write(safe_message + '\n')
            self.terminal.flush()
        self.logfile.write(message + '\n')
        self.logfile.flush()

    def close(self):
        self.logfile.close()

# --- Configuration ---
RULES_FILE_PATH = "rules.json"
JSON_FILE_PATH = "stock_history.json"
ORATS_FOLDER = "ORATS_json"

def load_and_run_simulation(rules_file_path, json_file_path):
    """
    Loads rules and data, sets up logging, and calls the core simulation engine.
    """
    print(f"Start Simulation: Loading rules from '{rules_file_path}'")

    # --- LOGGING SETUP ---
    try:
        with open(rules_file_path, 'rb') as f:
            rules_preview = orjson.loads(f.read())
            minimal_mode = bool(rules_preview.get("account_simulation", {}).get("Minimal_Print_Out", False))
    except Exception:
        minimal_mode = False
    
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(LOG_DIR, f"simulation_log_{timestamp}.log")
    
    original_stdout = sys.stdout
    logger = Logger(log_file_path, minimal_mode=minimal_mode)
    sys.stdout = logger

    try:
        # --- Load Rules and Stock History Data ---
        with open(rules_file_path, 'rb') as f:
            rules = orjson.loads(f.read())
        
        with open(json_file_path, 'rb') as f:
            stock_history_dict = orjson.loads(f.read())

        # --- Call the Core Simulation Engine ---
        run_simulation_in_memory(
            rules=rules,
            stock_history_dict=stock_history_dict,
            orats_folder=ORATS_FOLDER,
            logger=logger
        )

    finally:
        # --- Restore Standard Output and Close Logger ---
        sys.stdout = original_stdout
        logger.close()
        print(f"\nSimulation complete. Log saved to: {log_file_path}")

# --- Main Execution ---
if __name__ == "__main__":
    load_and_run_simulation(RULES_FILE_PATH, JSON_FILE_PATH)