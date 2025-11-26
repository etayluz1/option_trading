import json
import os

# File paths
input_path = r'c:\option_trading\ORATS_json\2025-11-11.json'
output_path = r'c:\option_trading\ORATS_json\2025-11-11.arr.json'

# The order: [strike, pBidPx, pAskPx, putDelta]
def option_to_array(opt):
    try:
        strike = float(opt.get('strike', 0))
        bid = float(opt.get('pBidPx', 0))
        ask = float(opt.get('pAskPx', 0))
        delta = float(str(opt.get('putDelta', '0')).replace('%',''))
        return [strike, bid, ask, delta]
    except Exception:
        return []

def process(obj):
    if isinstance(obj, dict):
        # If this dict looks like an option contract, convert it
        if all(k in obj for k in ('strike','pBidPx','pAskPx','putDelta')):
            return option_to_array(obj)
        # Otherwise, process recursively
        return {k: process(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [process(i) for i in obj]
    else:
        return obj

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

arr_data = process(data)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(arr_data, f, separators=(',', ':'))

orig_size = os.path.getsize(input_path)
arr_size = os.path.getsize(output_path)

print(f'Original size: {orig_size} bytes')
print(f'Array format size: {arr_size} bytes')
print(f'Reduction: {orig_size - arr_size} bytes ({(orig_size - arr_size) / orig_size * 100:.2f}%)')
