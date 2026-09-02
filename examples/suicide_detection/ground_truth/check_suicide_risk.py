"""
Script to check for suicide risk categories or related keywords in CounselChat dataset.
"""
from datasets import load_dataset
import pandas as pd
import sys

# Set encoding for output
sys.stdout.reconfigure(encoding='utf-8')

def check_suicide_risk():
    print("Loading full CounselChat dataset to check for suicide risk categories...")
    try:
        ds = load_dataset("nbertagnolli/counsel-chat")
        df = ds['train'].to_pandas()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check unique topics
    print(f"\n=== Topics Analysis ===")
    topics = df['topic'].unique()
    print(f"All available topics ({len(topics)}):")
    print(topics)
    
    # Check for specific suicide-related topics
    suicide_topics = [t for t in topics if 'suicide' in str(t).lower() or 'harm' in str(t).lower()]
    
    if suicide_topics:
        print(f"\nFound potential suicide-related topics: {suicide_topics}")
        for topic in suicide_topics:
            count = len(df[df['topic'] == topic])
            print(f"- Topic '{topic}': {count} entries")
    else:
        print("\nNo explicit 'suicide' or 'risk' category found in 'topic' column.")

    # Keyword search as proxy since no explicit label exists
    print(f"\n=== Keyword Search (Proxy for Risk) ===")
    keywords = ['suicid', 'kill myself', 'end my life', 'want to die', 'harm myself']
    
    for kw in keywords:
        # Check in titles and text
        matches = df[
            df['questionTitle'].fillna('').str.contains(kw, case=False) | 
            df['questionText'].fillna('').str.contains(kw, case=False) |
            df['topic'].fillna('').str.contains(kw, case=False)
        ]
        unique_matches = matches['questionID'].nunique()
        print(f"Keyword '{kw}': found in {unique_matches} unique questions")
        
        if unique_matches > 0:
            example = matches.iloc[0]
            print(f"  Example: {example['questionTitle']}")

if __name__ == "__main__":
    check_suicide_risk()
