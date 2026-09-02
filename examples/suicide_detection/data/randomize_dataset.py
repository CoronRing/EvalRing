
import csv
import random
from pathlib import Path
from collections import Counter
import sys

def main():
    script_dir = Path(__file__).resolve().parent
    main_csv = script_dir / 'res_15k_no_duplication.csv'
    out_csv = script_dir / 'res_15k_10k_random.csv'

    if not main_csv.exists():
        print(f'Error: {main_csv} not found.')
        sys.exit(1)

    print(f'Reading {main_csv}...')
    
    with open(main_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = list(reader)
        
    print(f'Original size: {len(lines)} rows.')
    
    user_idx = header.index('users') if 'users' in header else 1
    
    # Count posts per user
    user_counts = Counter(row[user_idx] for row in lines)
    
    # Filter out users with >= 50 posts
    filtered_lines = [row for row in lines if user_counts[row[user_idx]] < 50]
    
    print(f'Size after filtering users with >= 50 posts: {len(filtered_lines)} rows.')
    
    random.seed(42)
    random.shuffle(filtered_lines)
    
    sampled_lines = filtered_lines[:10000]
    
    print(f'Randomly selected {len(sampled_lines)} rows.')
    
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sampled_lines)
        
    print(f'Saved 10k random samples to {out_csv}')

if __name__ == '__main__':
    main()

