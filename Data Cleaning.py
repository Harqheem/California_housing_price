import pandas as pd
import numpy as np

def lines():
    '''Creates Line breaks for improved readability'''
    print('-' * 150)

if __name__ == '__main__':
    # Get Dataset Information
    df = pd.read_csv('housing.csv')
    df.info()
    lines()

    # Check for number of features and Observations
    observations, features = df.shape
    print(f"The dataset contains {features} features and {observations} observations ")
    lines()

    # Review data summary
    # print(df.describe)

    # Check for observation with missing values
    df_missing = df[df.isna().any(axis=1)]
    print(df_missing)
    lines()

    # After inspection. 207 observations were missing total_bedrooms information. So, they would be removed from the dataset
    df.dropna(inplace=True)

    # Check for duplicates
    df_duplicates = df[df.duplicated(keep=False)]
    print(df_duplicates)
    lines()

    # Get the observations with the maximum and minimim features
    max_feature = df.idxmax(numeric_only=True)
    min_feature = df.idxmin(numeric_only=True)

    print("Maximum Features and Values\n")
    print(f"{'Features':<25} {'Value':<25} {'Observation Number':<25}")
    for col_name, row_idx in max_feature.items():
        max_val = df.at[row_idx, col_name]
        print(f"{col_name:<25} {max_val:<25} {row_idx:<25}")

    print("\nMinimum Features and Values\n")
    print(f"{'Features':<25} {'Value':<25} {'Observation Number':<25}")
    for col_name, row_idx in min_feature.items():
        min_val = df.at[row_idx, col_name]
        print(f"{col_name:<25} {min_val:<25}  {row_idx:<25}")

    lines()

    # Export dataset for model training
    df.to_csv('housing_cleaned.csv')