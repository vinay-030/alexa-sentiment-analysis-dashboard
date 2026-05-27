import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("==================================================")
    print("STEP 1 — DATASET VALIDATION")
    print("==================================================")
    
    try:
        df = pd.read_csv('amazon_alexa.tsv', sep='\t')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Print requested information
    print(f"Total rows: {len(df)}")
    print(f"Null values:\n{df.isnull().sum()}\n")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Unique sentiment labels: {df['feedback'].unique()}\n")
    
    # Class imbalance check
    print("Sentiment Distribution (feedback column):")
    distribution = df['feedback'].value_counts()
    print(distribution)
    print()
    
    positive_count = distribution.get(1, 0)
    negative_count = distribution.get(0, 0)
    total_count = positive_count + negative_count
    
    print(f"Positive (1): {positive_count} ({positive_count/total_count*100:.2f}%)")
    print(f"Negative (0): {negative_count} ({negative_count/total_count*100:.2f}%)")
    
    if positive_count > negative_count * 5 or negative_count > positive_count * 5:
         print("WARNING: Severe class imbalance detected! Must use class_weight='balanced' or sampling.")
    else:
         print("Dataset is relatively balanced.")

    # Generate and save plot
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='feedback', palette=['#ff6b6b', '#1dd1a1'])
    plt.title('Sentiment Distribution (0 = Negative, 1 = Positive)')
    plt.xlabel('Sentiment Label')
    plt.ylabel('Number of Reviews')
    plt.savefig('sentiment_distribution.png', bbox_inches='tight')
    plt.close()
    print("\nSaved sentiment distribution graph to 'sentiment_distribution.png'")

if __name__ == "__main__":
    main()
