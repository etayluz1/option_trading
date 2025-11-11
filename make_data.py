import os, csv, re, json
from datetime import datetime

# Run make_data.py (It will read ORATS_csv and generate c:\option_trading\ORATS_json)

# Configuration (Copied from User Input) ---
csv_folder = "D:\\ORATS_csv"
json_folder = "c:\\option_trading\\ORATS_json"    # New JSON folder
columns_to_keep = ["expirDate", "strike", "pBidPx", "pAskPx", "delta"]

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
        quit()
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

# Note: Configuration is set at the top of the file

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
                    continue
            sp500_dict[date_obj] = set(t.strip() for t in tickers if t.strip())

def process_csv(stop_after_first=False):
    """
    Processes the raw ORATS CSV files, filters by S&P 500/ETFs, applies a pBidPx filter, 
    calculates putDelta as a percentage string rounded to 8 decimal places,
    and outputs a single JSON file per day, overriding existing ones.
    """
    print("\n=== Starting CSV Processing ===")
    print(f"CSV Folder: {csv_folder}")
    print(f"JSON Folder: {json_folder}")
    print(f"Looking for SP500 file: {sp500_file}")

    if not os.path.exists(csv_folder):
        print(f"ERROR: CSV folder not found at {csv_folder}")
        return

    load_sp500_data()
    print(f"Loaded SP500 data: {len(sp500_dict)} dates found")
    
    os.makedirs(json_folder, exist_ok=True)
    print(f"Created/verified JSON folder: {json_folder}")
    
    # option_fields_to_keep ensures the raw data is kept, excluding the date/ticker fields
    option_fields_to_keep = [col for col in columns_to_keep if col not in ["expirDate", "ticker"]]
    
    # Recursively discover CSV files inside the csv_folder directory and
    # keep relative paths so nested files can be processed and their
    # filenames (used for date extraction) remain available.
    files_to_process = []
    print("\nScanning for CSV files...")
    for root, _, files in os.walk(csv_folder):
        for fname in files:
            if fname.lower().endswith(".csv"):
                rel_path = os.path.relpath(os.path.join(root, fname), csv_folder)
                files_to_process.append(rel_path)
                print(f"Found CSV: {rel_path}")
    files_to_process = sorted(files_to_process)
    print(f"\nTotal CSV files found: {len(files_to_process)}")
    processed_count = 0

    for file in files_to_process:
        input_path = os.path.join(csv_folder, file)

        match = re.search(r"(\d{8})", file)
        if not match:
            print(f"Skipping file {file}: No date found in filename.")
            continue
        
        file_date_str_yyyymmdd = match.group(1)
        file_date_dt = datetime.strptime(file_date_str_yyyymmdd, "%Y%m%d").date()
        
        output_filename = f"{file_date_dt.strftime('%Y-%m-%d')}.json"
        output_path = os.path.join(json_folder, output_filename)

        
        # --- NEW: If the JSON already exists, skip processing to avoid overwriting ---
        if os.path.exists(output_path):
            print(f"⏩ Skipping {file}: Output JSON '{output_path}' already exists. Not overriding.")
            continue

        print(f"\n=== Processing file: {file} ===")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        
        sp500_dates = sorted(sp500_dict.keys())
        relevant_date = max((d for d in sp500_dates if d <= file_date_dt), default=None)
        print(f"File date: {file_date_dt}, Relevant SP500 date: {relevant_date}")
        
        if relevant_date is None and sp500_dict:
             print(f"Skipping file {file}: No relevant S&P 500 list found for {file_date_dt} (before or on file date).")
             continue
        
        sp500_tickers = sp500_dict.get(relevant_date, set()) if relevant_date else set()
        
        daily_options_data = {}

        print(f"Processing {file}...")

        try:
            with open(input_path, newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                
                ticker_field = reader.fieldnames[0] if reader.fieldnames else None
                print(f"CSV Headers found: {reader.fieldnames}")
                print(f"Using ticker field: {ticker_field}")
                
                if not ticker_field:
                    print(f"Skipping file {file}: Could not determine ticker field.")
                    continue
                
                for row in reader:
                    symbol = row.get(ticker_field)
                    
                    # --- Filter pBidPx == "0.00" ---
                    if row.get("pBidPx") == "0.00":
                        continue
                    # --------------------------------

                    if symbol in sp500_tickers or symbol in etfs:
                        expir_date_str = row.get("expirDate")
                        if not expir_date_str:
                            continue
                        
                        try:
                            expir_date_dt = datetime.strptime(expir_date_str, "%m/%d/%Y").date()
                        except ValueError:
                            continue

                        days_interval = (expir_date_dt - file_date_dt).days
                        
                        # Create the filtered option record with raw data
                        option_record = {col: row.get(col) for col in option_fields_to_keep}
                        
                        # 🚨 CHANGE 2: Calculate and format 'putDelta' as percentage string
                        delta_str = row.get("delta")
                        if delta_str is not None:
                            try:
                                delta_float = float(delta_str)
                                
                                # 1. Calculate Put Delta: delta - 1.0 (assuming 'delta' is the call delta)
                                put_delta_raw = delta_float - 1.0
                                
                                # 2. Round to 4 decimal places
                                put_delta_rounded = round(put_delta_raw, 4)
                                
                                # 3. Convert to percentage string (multiplied by 100)
                                put_delta_percent_str = f"{put_delta_rounded * 100.0:.2f}%"
                                
                                option_record["putDelta"] = put_delta_percent_str

                            except ValueError:
                                # Handle case where delta is not a valid number
                                option_record["putDelta"] = None
                        else:
                            option_record["putDelta"] = None
                        # --------------------------------------------------
                        
                        # Initialize structures
                        if symbol not in daily_options_data:
                            daily_options_data[symbol] = {}
                        
                        if expir_date_str not in daily_options_data[symbol]:
                            daily_options_data[symbol][expir_date_str] = {
                                "days_interval": days_interval, 
                                "options": []                   
                            }
                        
                        option_record["strike"] = float(option_record["strike"]) # Convert strike to float for numerical consistency
                        
                        # Append the option record
                        daily_options_data[symbol][expir_date_str]["options"].append(option_record)

            print(f"\nProcessed data summary:")
            print(f"Total symbols processed: {len(daily_options_data)}")
            for symbol in daily_options_data:
                print(f"- {symbol}: {sum(len(exp['options']) for exp in daily_options_data[symbol].values())} options")
            
            # Write the aggregated data to the JSON file
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(daily_options_data, outfile, indent=4)

            print(f"✅ Successfully processed and SAVED/OVERRODE JSON file to: {output_path}")
            processed_count += 1
            print(f"Total files processed so far: {processed_count}")
            
            if stop_after_first:
                print("Stopping after processing the first file (stop_after_first=True).")
                break

        except Exception as e:
            print(f"🛑 Error processing file {file}: {e}")
            import traceback
            traceback.print_exc()
            continue

# --- Execution ---
# To process all files: process_csv()
# To stop after the first file (for debugging): process_csv(stop_after_first=True)
process_csv(stop_after_first=False)