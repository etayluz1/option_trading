import os
import zipfile
import csv
import io # Used to read text from zip file contents (in memory)

# --- Configuration ---
extract_to = "ORATS_csv"
columns_to_keep = [
    "ticker", "expirDate", "strike", "pVolu", "pOi", 
    "pBidPx", "pAskPx", "pMidIv", "delta"
]
# ---------------------


def extract_and_filter_zip():
    """
    Extracts zip files from 'ORATS_zip', reads the contained CSV files, 
    filters them to keep only specified columns AND exclude rows where
    pBidPx is 0, and writes the filtered CSV to the 'ORATS_csv' directory.
    """
    folder_path = "ORATS_zip"
    
    # 1. Check if the input folder exists
    if not os.path.exists(folder_path):
        print(f"🛑 Error: Input folder '{folder_path}' not found. Cannot proceed with extraction.")
        return

    # Ensure the output directory exists
    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Starting zip extraction and CSV filtering into '{extract_to}'...")
    
    files_processed = 0

    # Walk the input folder recursively so we handle nested directories of zip files
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith(".zip"):
                continue

            zip_path = os.path.join(root, filename)
            # Preserve relative path under the output directory to avoid name collisions
            rel_dir = os.path.relpath(root, folder_path)
            out_dir_for_zip = os.path.join(extract_to, rel_dir) if rel_dir != '.' else extract_to
            os.makedirs(out_dir_for_zip, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    
                    # Assume each zip contains one CSV file
                    csv_names = [n for n in zip_ref.namelist() if n.endswith('.csv')]
                    
                    if not csv_names:
                        print(f"⚠️ Warning: Zip file '{zip_path}' contains no CSV files. Skipping.")
                        continue

                    # Process the first CSV file found in the zip
                    csv_in_zip_name = csv_names[0]
                    
                    # 1. Read the CSV content from the zip file object
                    with zip_ref.open(csv_in_zip_name, 'r') as csv_file_in_zip:
                        
                        # Use io.TextIOWrapper to read the binary data as text (assuming utf-8)
                        csv_content = io.TextIOWrapper(csv_file_in_zip, encoding='utf-8')
                        reader = csv.DictReader(csv_content)
                        
                        # Determine the output filename (use the name of the CSV within the zip)
                        # and place it inside the corresponding relative output directory
                        output_csv_path = os.path.join(out_dir_for_zip, csv_in_zip_name)

                        # If the CSV already exists, skip processing to avoid overwriting
                        if os.path.exists(output_csv_path):
                            print(f"⏩ Skipping existing file '{output_csv_path}' — already extracted.")
                            continue

                        # 2. Write the filtered content to the output path
                        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
                            writer = csv.DictWriter(
                                outfile, 
                                fieldnames=columns_to_keep, 
                                extrasaction='ignore'
                            )
                            writer.writeheader()
                            
                            rows_written = 0
                            rows_skipped_zero_bid = 0
                            
                            for row in reader:
                                # --- NEW FILTERING LOGIC ---
                                # Check if 'pBidPx' exists and if its value (converted to float) is NOT 0
                                # Use a try-except block for robust handling of missing or non-numeric values
                                try:
                                    bid_price = float(row.get('pBidPx', 0))
                                    if bid_price != 0.0:
                                        # The DictWriter automatically filters the columns
                                        writer.writerow(row)
                                        rows_written += 1
                                    else:
                                        rows_skipped_zero_bid += 1
                                except (ValueError, TypeError):
                                    # Handle cases where pBidPx is non-numeric or malformed
                                    rows_skipped_zero_bid += 1 # Treat as a skipped row
                                # ---------------------------

                print(f"✅ Extracted and Filtered: '{csv_in_zip_name}' from '{filename}' ({rows_written} rows written, {rows_skipped_zero_bid} skipped due to zero/invalid bid).")
                files_processed += 1
                
            except zipfile.BadZipFile:
                print(f"❌ Error: '{filename}' is a corrupt or unreadable zip file.")
            except Exception as e:
                print(f"❌ Error processing '{filename}': {e}")
        
    if files_processed == 0:
        print("⚠️ No zip files found in 'ORATS_zip' to extract and filter.")
    else:
        print(f"Finished extraction and filtering. Total zips processed: {files_processed}.")


# --- Execution ---
extract_and_filter_zip()