import pandas as pd
from load_data_ai import df_ai

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by removing duplicates and handling missing values.

    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values by dropping rows with any missing values
    df = df.dropna().reset_index(drop=True)

    return df

preprocessed_ai_df = preprocess_data(df_ai)

print(preprocessed_ai_df.head())
print(f"Number of records after preprocessing: {len(preprocessed_ai_df)}")
