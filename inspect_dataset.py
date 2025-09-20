import os
import pandas as pd

def count_images(directory):
    """Counts the number of images in each class subdirectory."""
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts

def profile_datasets(train_dir, test_dir):
    """Profiles the training and testing datasets and displays the results."""
    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)

    df = pd.DataFrame({
        'Train': pd.Series(train_counts),
        'Test': pd.Series(test_counts)
    }).fillna(0).astype(int)

    # Calculate totals and ratios
    df.loc['Total'] = df.sum()
    df['Train_Ratio'] = (df.loc[df.index != 'Total', 'Train'] / df.loc['Total', 'Train']) * 100
    df['Test_Ratio'] = (df.loc[df.index != 'Total', 'Test'] / df.loc['Total', 'Test']) * 100
    
    print("Dataset Profile:")
    print(df.to_string(float_format="%.2f%%"))
    print("\n--- Analysis ---")

    # Analyze Test Set
    if df.loc['Total', 'Test'] < 500:
        print(f"⚠️  Warning: The total number of test images ({df.loc['Total', 'Test']}) is quite small. Metrics may be noisy.")
    
    min_test_class = df['Test'].drop('Total').idxmin()
    min_test_count = df.loc[min_test_class, 'Test']

    if min_test_count < 30:
        print(f"⚠️  Warning: The class '{min_test_class}' has only {min_test_count} test images. This is very low and can lead to unstable accuracy for this class.")
    
    # Analyze Imbalance
    test_ratios = df['Test_Ratio'].dropna()
    if test_ratios.max() / test_ratios.min() > 5:
        print("⚠️  Warning: The test set appears to be significantly imbalanced. The model may perform poorly on minority classes.")


if __name__ == '__main__':
    train_directory = "data/processed/train"
    test_directory = "data/processed/test"
    profile_datasets(train_directory, test_directory)