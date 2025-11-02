import csv
import os
from datetime import datetime

def export_trades_to_csv(closed_trades_log, output_dir="excel_exports"):
    """Export closed trades to a CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"trades_{timestamp}.csv")
    
    # Define the columns for the trades CSV
    fieldnames = [
        'Exit #',
        'Ticker',
        'Qty',
        'Day In',
        'Price In',
        'Amount In',
        'Day Out',
        'Price Out',
        'Amount Out',
        'Reason Why Closed',
        'Gain $',
        'Gain %'
    ]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for index, trade in enumerate(closed_trades_log, 1):
            row = {
                'Exit #': index,
                'Ticker': trade['Ticker'],
                'Qty': trade['Qty'],
                'Day In': trade['DayIn'],
                'Price In': trade['PriceIn'],
                'Amount In': trade['AmountIn'],
                'Day Out': trade['DayOut'],
                'Price Out': trade['PriceOut'],
                'Amount Out': trade['AmountOut'],
                'Reason Why Closed': trade['ReasonWhyClosed'],
                'Gain $': trade['Gain$'],
                'Gain %': trade['Gain%']
            }
            writer.writerow(row)
    
    return filename

def export_monthly_performance_to_csv(monthly_performance, output_dir="excel_exports"):
    """Export monthly performance to a CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"monthly_performance_{timestamp}.csv")
    
    # Define the columns for the monthly performance CSV
    fieldnames = [
        'Month',
        'Total Value End',
        '$ Gain',
        '% Gain',
        '% SPY Gain'
    ]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for (year, month), data in sorted(monthly_performance.items()):
            month_label = datetime(year, month, 1).strftime('%Y-%m')
            row = {
                'Month': month_label,
                'Total Value End': data['end_value'],
                '$ Gain': data['gain_abs'],
                '% Gain': data['gain_pct'],
                '% SPY Gain': data['spy_gain_pct']
            }
            writer.writerow(row)
    
    return filename