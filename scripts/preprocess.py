import pandas as pd

def clean_dataset(input_file="c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/data/Tweets.csv"):
    # Load dataset first
    df = pd.read_csv(input_file)

    # Print column names to debug issues
    print("Columns in dataset:", df.columns)

    # Ensure 'text' column exists
    if 'text' not in df.columns:
        raise KeyError("The column 'text' is missing in the dataset! Check CSV file headers.")

    # Remove rows with missing values in 'text' column
    df = df.dropna(subset=['text'])  

    # Convert text to lowercase (example preprocessing)
    df['cleaned_text'] = df['text'].astype(str).apply(lambda x: x.lower())

    # Save cleaned data
    output_file = "c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/data/cleaned_airline_tweets.csv"
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    clean_dataset()
