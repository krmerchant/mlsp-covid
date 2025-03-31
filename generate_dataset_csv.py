import pandas as pd
from sklearn.model_selection import train_test_split
import click


@click.command()
@click.argument("csvfile")
def partition_dataset(csvfile:str):

    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csvfile)  # Replace 'your_file.csv' with your actual file name
    
    # Perform a stratified split to ensure even distribution of categorys
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['category'], random_state=42)
    
    # Save the train and test data to separate CSV files
    train_df.to_csv('train_file.csv', index=False)
    test_df.to_csv('test_file.csv', index=False)
    
    # Extra step: Read the saved files and verify stratification
    train_check = pd.read_csv('train_file.csv')
    test_check = pd.read_csv('test_file.csv')
    
    # Calculate and print the category distributions for the original, train, and test data
    print("Original dataset category distribution:")
    print(df['category'].value_counts(normalize=True))
    print(df['category'].value_counts())
    
    print("\nTrain dataset category distribution:")
    print(train_check['category'].value_counts(normalize=True))
    print(train_check['category'].value_counts())
    
    print("\nTest dataset category distribution:")
    print(test_check['category'].value_counts(normalize=True))
    print(test_check['category'].value_counts())
    
    print("\nTrain and Test files have been created with an 80/20 split.")
