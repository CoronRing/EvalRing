
import csv
import argparse
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Visualize Suicide Detection Dataset')
    parser.add_argument('--input', type=str, default='res_15k_10k_random.csv', help='Input CSV file')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    input_path = script_dir / args.input
    media_dir = script_dir / 'media'
    media_dir.mkdir(exist_ok=True)

    if not input_path.exists():
        print(f'Error: {input_path} not found.')
        return

    users = []
    word_counts = []
    sentiments = []

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user = row.get('users', row.get('user', ''))
            users.append(user)
            
            text = row.get('text', '')
            words = len(text.split())
            word_counts.append(words)
            
            sentiment = row.get('sentiment', 'Unknown')
            sentiments.append(sentiment)

    total_posts = len(users)
    unique_users = len(set(users))
    avg_posts = total_posts / unique_users if unique_users else 0
    avg_words = sum(word_counts) / total_posts if total_posts else 0
    
    sorted_words = sorted(word_counts)
    median_words = sorted_words[total_posts // 2] if total_posts else 0

    print(f'--- Dataset Statistics ({input_path.name}) ---')
    print(f'Total Posts: {total_posts}')
    print(f'Total Unique Users: {unique_users}')
    print(f'Average Posts per User: {avg_posts:.2f}')
    print(f'Average Word Count: {avg_words:.2f} words')
    print(f'Median Word Count: {median_words} words')
    
    print('\nLabel Distribution')
    sentiment_counts = Counter(sentiments)
    for k, v in sentiment_counts.most_common():
        print(f'{k}: {v}')

    user_counts = list(Counter(users).values())
    plt.figure(figsize=(10, 6))
    max_uc = max(user_counts) if user_counts else 1
    plt.hist(user_counts, bins=range(1, max_uc + 5), color='skyblue', edgecolor='black')
    plt.title('Posts per User Distribution')
    plt.xlabel('Number of Posts')
    plt.ylabel('Frequency (Users)')
    plt.tight_layout()
    plt.savefig(media_dir / 'posts_per_user.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Post Lengths (Words)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency (Posts)')
    plt.tight_layout()
    plt.savefig(media_dir / 'post_lengths.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sents, counts = zip(*sentiment_counts.most_common())
    plt.bar(sents, counts, color='salmon', edgecolor='black')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Posts')
    plt.tight_layout()
    plt.savefig(media_dir / 'sentiment_dist.png')
    plt.close()

    print(f'\nVisualizations saved to {media_dir}')

if __name__ == '__main__':
    main()

