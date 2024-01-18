import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads in the datasets and merges them into one DataFrame.

    Args:
        messages_filepath (str): File path to the messages dataset
        categories_filepath (str): File path to the categories dataset
    
    Returns:
        pd.DataFrame: Merged DataFrame containing both datasets
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on="id")


def clean_data(df):
    """
    Cleans the DataFrame by expanding the categories, removing irrelevant columns and dropping duplicate rows.

    Args:
        df (pd.DataFrame): DataFrame to be cleaned
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """

    # Split the categories column into all of it's features 
    categories = df['categories'].str.split(";", expand=True)

    # Set the column names for the new categories based on the first row
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x.strip("-10"))
    categories.columns = category_colnames

    for column in categories:
        # Get the value of the row for each category
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])


    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df.drop("original", axis=1, inplace=True)

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the data into a SQL database.

    Args:
        df (pd.DataFrame): DataFrame to be saved
        database_filename (string): Database name to be saved to
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()