import os, csv, re, json
from datetime import datetime

# --- Configuration (Copied from User Input) ---
extract_to = "ORATS_csv"
columns_to_keep = ["expirDate", "strike", "pVolu", "pOi", "pBidPx", "pAskPx", "pMidIv", "delta"]
processed_folder_csv = "2025_processed" # Original CSV folder (Unused in this function)
processed_folder_json = "ORATS_json"    # New JSON folder

sp500_file = "SP500_history.csv"
sp500_dict = {}
etfs = [
    "ACWI", "ACWX", "AGG", "AIQ", "AVLV", "BBAX", "BBEU", "BBIN", "BBJP", "BBUS",
    "BIL", "BND", "BNDX", "BSV", "BUFR", "CGGO", "CGGR", "DFAC", "DFIV", "DGRO",
    "DGRW", "DIA", "DIVO", "DLN", "DUHP", "DYNF", "EEM", "EFA", "EFAV", "EFG",
    "EFV", "EMXC", "ESGD", "ESGE", "ESGU", "ESGV", "EWJ", "EWT", "EWY", "EWZ",
    "EZU", "FDL", "FDN", "FDVV", "FNDE", "FNDF", "FNDX", "FTCS", "FTEC", "FXI",
    "GLD", "GSLC", "GUNR", "HDV", "HEFA", "IAU", "IBIT", "IDEV", "IEF", "IEFA",
    "IEMG", "IEUR", "IGM", "IGV", "IJH", "IJR", "INDA", "IQLT", "ITOT", "IUSB",
    "IUSG", "IUSV", "IVE", "IVV", "IVW", "IWB", "IWD", "IWF", "IWM", "IWR", "IWV",
    "IWY", "IXN", "IXUS", "IYW", "JEPI", "JEPQ", "JGRO", "JPST", "JQUA", "MBB",
    "MCHI", "MTUM", "MUB", "NOBL", "ONEQ", "PBUS", "PRF", "QLD", "QQQ", "QQQI",
    "QQQM", "QUAL", "QYLD", "RDVY", "RSP", "RWL", "SCHB", "SCHD", "SCHE", "SCHF",
    "SCHG", "SCHV", "SCHX", "SGOV", "SMH", "SOXL", "SOXX", "SPDW", "SPEM", "SPHQ",
    "SPLG", "SPLV", "SPMO", "SPTM", "SPXL", "SPY", "SPYD", "SPYG", "SPYI", "SPYV",
    "SSO", "TLT", "TQQQ", "TSLL", "URTH", "VB", "VCIT", "VCR", "VCSH", "VDC",
    "VEA", "VEU", "VFH", "VGIT", "VGK", "VGT", "VHT", "VIG", "VIGI", "VLUE", "VNQ",
    "VO", "VONE", "VONG", "VONV", "VOO", "VOOG", "VOOV", "VOX", "VPL", "VSGX", "VT",
    "VTEB", "VTI", "VTV", "VUG", "VV", "VWO", "VXUS", "VYM", "VYMI", "XLB", "XLC",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"
]
# ---------------------------------------------

def load_sp500_data():
    """Loads S&P 500 historical component data from sp500_file into sp500_dict."""
    global sp500_dict
    if not os.path.exists(sp500_file):
         # Placeholder for case where the file is missing
        print(f"Warning: S&P 500 history file '{sp500_file}' not found. Filtering will only use the hardcoded ETFs list.")
        return
        
    with open(sp500_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row: continue # Skip empty rows
            date_str = row[0].strip()
            tickers = row[1].split(',') if len(row) > 1 else []
            try:
                # Assuming the format is consistently M/D/YY or M/DD/YY
                date_obj = datetime.strptime(date_str, "%m/%d/%y").date()
            except ValueError:
                # Try YYYY format if possible
                try:
                    date_obj = datetime.strptime(date_str, "%m/%d/%Y").date()
                except ValueError:
                    # Skip row if date format is unreadable
                    # print(f"Warning: Could not parse date '{date_str}' in {sp500_file}.")
                    continue
            sp500_dict[date_obj] = set(t.strip() for t in tickers if t.strip())

def process_csv(stop_after_first=False):
    """
    Processes the raw ORATS CSV files, filters by S&P 500/ETFs, applies a pBidPx filter, 
    and outputs a single JSON file per day, overriding existing ones.
    """

    load_sp500_data()
    os.makedirs(processed_folder_json, exist_ok=True)
    
    # Fields for the individual option record (excluding 'ticker' and 'expirDate')
    option_fields_to_keep = [col for col in columns_to_keep if col not in ["expirDate", "ticker"]]
    
    files_to_process = sorted([f for f in os.listdir(extract_to) if f.endswith(".csv")])
    processed_count = 0

    for file in files_to_process:
        input_path = os.path.join(extract_to, file)

        match = re.search(r"(\d{8})", file)
        if not match:
            print(f"Skipping file {file}: No date found in filename.")
            continue
        
        file_date_str_yyyymmdd = match.group(1)
        # 1. Parse the trade/file date (YYYY-MM-DD)
        file_date_dt = datetime.strptime(file_date_str_yyyymmdd, "%Y%m%d").date()
        
        # New filename format: YYYY-MM-DD.json
        output_filename = f"{file_date_dt.strftime('%Y-%m-%d')}.json"
        output_path = os.path.join(processed_folder_json, output_filename)

        # Find relevant S&P 500 tickers
        sp500_dates = sorted(sp500_dict.keys())
        # Find the latest S&P 500 list from *before or on* the file date
        relevant_date = max((d for d in sp500_dates if d <= file_date_dt), default=None)
        
        if relevant_date is None and sp500_dict: # Check if dict is not empty but no date match
             print(f"Skipping file {file}: No relevant S&P 500 list found for {file_date_dt} (before or on file date).")
             continue
        
        sp500_tickers = sp500_dict.get(relevant_date, set()) if relevant_date else set()
        
        # Daily options data structure: {ticker: {expirDate: {"days_interval": int, "options": [...]}}}
        daily_options_data = {}

        print(f"Processing {file}...")

        try:
            with open(input_path, newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                
                # Dynamic Ticker Field: Assume the first field is the ticker symbol
                ticker_field = reader.fieldnames[0] if reader.fieldnames else None
                if not ticker_field:
                    print(f"Skipping file {file}: Could not determine ticker field.")
                    continue
                
                for row in reader:
                    symbol = row.get(ticker_field)
                    
                    # --- NEW REQUIREMENT: Filter pBidPx == "0.00" ---
                    if row.get("pBidPx") == "0.00":
                        continue
                    # --------------------------------------------------

                    # Filter by S&P 500 or ETF list
                    if symbol in sp500_tickers or symbol in etfs:
                        expir_date_str = row.get("expirDate")
                        if not expir_date_str:
                            continue
                        
                        # 2. Parse the expiration date (M/D/YYYY)
                        try:
                            expir_date_dt = datetime.strptime(expir_date_str, "%m/%d/%Y").date()
                        except ValueError:
                            # Skip row if date format is incorrect
                            print(f"Warning: Could not parse expiration date '{expir_date_str}' in file {file}.")
                            continue

                        # 3. Calculate days interval
                        days_interval = (expir_date_dt - file_date_dt).days
                        
                        # Create the filtered option record
                        option_record = {col: row.get(col) for col in option_fields_to_keep}
                        
                        # Initialize structures
                        if symbol not in daily_options_data:
                            daily_options_data[symbol] = {}
                        
                        if expir_date_str not in daily_options_data[symbol]:
                            # Initialize the dictionary with 'days_interval'
                            daily_options_data[symbol][expir_date_str] = {
                                "days_interval": days_interval, 
                                "options": []                   
                            }
                        
                        # 4. Append the option record to the "options" list
                        daily_options_data[symbol][expir_date_str]["options"].append(option_record)

            # Write the aggregated data to the JSON file (Overriding existing file)
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(daily_options_data, outfile, indent=4)

            print(f"✅ Successfully processed and SAVED/OVERRODE JSON file to: {output_path}")        
            processed_count += 1
            
            # --- NEW REQUIREMENT: Stop after first file ---
            if stop_after_first:
                print("Stopping after processing the first file (stop_after_first=True).")
                break
            # -----------------------------------------------

        except Exception as e:
            print(f"🛑 Error processing file {file}: {e}")
            continue

# --- Execution ---
# To process all files: process_csv()
# To stop after the first file (for debugging): process_csv(stop_after_first=True)
process_csv(stop_after_first=True)