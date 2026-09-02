"""
Script to explore the RSD_15K (Reddit Suicide Detection) dataset.
Fixed schema matching and column detection.
"""
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set encoding for output
sys.stdout.reconfigure(encoding='utf-8')

def explore_rsd15k():
    csv_path = "rsd_15k.csv"
    print(f"Loading dataset from found csv {csv_path}...")
    
    # Try different encodings
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin1')
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return
    except Exception as e:
        print(f"Generic error loading CSV: {e}")
        return

    print(f"\n=== Basic Statistics ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify key columns based on content
    text_col = 'text' if 'text' in df.columns else df.columns[1] # usually second col
    label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[-1] # possibly last
    
    print(f"Identified Text Column: '{text_col}'")
    print(f"Identified Label Column: '{label_col}'")

    print(f"\n=== Label Analysis ('{label_col}') ===")
    if label_col in df.columns:
        counts = df[label_col].value_counts()
        print(counts)
        print("\nPercentage:")
        print((counts / len(df) * 100).round(2))
    else:
        print("Label column not found.")

    print(f"\n=== Text Analysis (Column: '{text_col}') ===")
    if text_col in df.columns:
        # Check if text column is actually text
        if df[text_col].dtype != object:
            df[text_col] = df[text_col].astype(str)
            
        df['word_count'] = df[text_col].fillna('').apply(lambda x: len(str(x).split()))
        print(df['word_count'].describe())
        
        print("\n--- Examples of 'suicide' class ---")
        suicide_indicators = [1, 'suicide', 'risk', 'high', 'Suicide']
        pos_label = None
        for ind in suicide_indicators:
            if ind in df[label_col].unique():
                pos_label = ind
                break
        
        if pos_label is not None:
             examples = df[df[label_col] == pos_label][text_col].head(3).tolist()
             for i, ex in enumerate(examples):
                print(f"\n[Example {i+1}]: {str(ex)[:200]}...")
        else:
             print("Could not isolate 'Suicide' class examples automatically.")

        # Check for anonymization
        print(f"\n=== Anonymization Check ===")
        normalization_pattern = r'\[USER\]|\[URL\]|\[NAME\]'
        matches = df[text_col].astype(str).str.contains(normalization_pattern, regex=True).sum()
        print(f"Rows containing tokens [USER], [URL], etc: {matches} ({matches/len(df)*100:.2f}%)")
    else:
        print("Text column not found.")

if __name__ == "__main__":
    explore_rsd15k()
