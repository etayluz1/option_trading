import json
import os

# File paths
input_path = r'c:\option_trading\ORATS_json\2025-11-11.json'
output_path = r'c:\option_trading\ORATS_json\2025-11-11.min.json'

# Key mapping for reduction
key_map = {
    'strike': 'S',
    'pBidPx': 'B',
    'pAskPx': 'A',
    'putDelta': 'D',
}

def replace_keys(obj):
    if isinstance(obj, dict):
        return {key_map.get(k, k): replace_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_keys(i) for i in obj]
    else:
        return obj

# Read original JSON
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Replace keys
min_data = replace_keys(data)

# Write minimized JSON (compact, no spaces)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(min_data, f, separators=(',', ':'))

# Get file sizes
orig_size = os.path.getsize(input_path)
min_size = os.path.getsize(output_path)

print(f'Original size: {orig_size} bytes')
print(f'Minimized size: {min_size} bytes')
print(f'Reduction: {orig_size - min_size} bytes ({(orig_size - min_size) / orig_size * 100:.2f}%)')
