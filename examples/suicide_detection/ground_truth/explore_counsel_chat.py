"""
Script to explore the full CounselChat dataset and report interesting findings.
"""
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set encoding for output
sys.stdout.reconfigure(encoding='utf-8')

def explore_dataset():
    print("Loading full CounselChat dataset from HuggingFace...")
    try:
        ds = load_dataset("nbertagnolli/counsel-chat")
        df = ds['train'].to_pandas()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\n=== Basic Statistics ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    unique_questions = df['questionID'].nunique()
    print(f"Unique Questions (by ID): {unique_questions}")
    print(f"Average answers per question: {len(df) / unique_questions:.2f}")

    print(f"\n=== Topic Analysis ===")
    topic_counts = df['topic'].value_counts()
    print(f"Total unique topics: {len(topic_counts)}")
    print("\nTop 10 Topics by frequency:")
    print(topic_counts.head(10))
    
    print("\nBottom 5 Topics:")
    print(topic_counts.tail(5))

    print(f"\n=== Length Analysis ===")
    df['question_len'] = df['questionText'].fillna('').apply(lambda x: len(str(x).split()))
    df['title_len'] = df['questionTitle'].fillna('').apply(lambda x: len(str(x).split()))
    df['answer_len'] = df['answerText'].fillna('').apply(lambda x: len(str(x).split()))
    
    print("Word Count Statistics:")
    print(df[['question_len', 'answer_len']].describe())
    
    print(f"\nLongest Answer: {df['answer_len'].max()} words")
    print(f"Shortest Answer: {df['answer_len'].min()} words")
    
    # Check short answers
    short_answers = df[df['answer_len'] < 10]['answerText'].head(3).tolist()
    print("\nExamples of very short answers (<10 words):")
    for ans in short_answers:
        print(f"- {ans}")

    print(f"\n=== Therapist Activity ===")
    therapist_counts = df['therapistURL'].value_counts()
    print(f"Total unique therapists: {len(therapist_counts)}")
    print("\nTop 5 Most Active Therapists:")
    print(therapist_counts.head(5))
    
    # Analyze upvotes
    print(f"\n=== Upvotes Analysis ===")
    print(df['upvotes'].describe())
    
    most_upvoted = df.loc[df['upvotes'].idxmax()]
    print(f"\nMost Upvoted Answer ({most_upvoted['upvotes']} upvotes):")
    print(f"Question: {most_upvoted['questionTitle']}")
    print(f"Topic: {most_upvoted['topic']}")
    print(f"Snippet: {most_upvoted['answerText'][:100]}...")

    # Correlation between length and upvotes
    corr = df['answer_len'].corr(df['upvotes'])
    print(f"\nCorrelation between answer length and upvotes: {corr:.4f}")

    # Finding: Questions with most answers
    answers_per_q = df['questionID'].value_counts()
    print(f"\n=== Question Popularity ===")
    print(f"Max answers for a single question: {answers_per_q.max()}")
    print("Top 3 most answered questions (IDs):")
    for qid in answers_per_q.head(3).index:
        q_data = df[df['questionID'] == qid].iloc[0]
        print(f"- ID {qid} ({answers_per_q[qid]} answers): {q_data['questionTitle']}")

if __name__ == "__main__":
    explore_dataset()
