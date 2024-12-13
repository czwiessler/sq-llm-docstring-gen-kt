import pandas as pd
from typing import Callable

def filter_files_based_on_conditions(metrics_file:str, conditions:dict[str, Callable], output_file:str) -> pd.DataFrame:
    """This function filterest the list of Python file names based on the provided conditions.

    Args:
        metrics_file (str): Filename of the .xlsx file containing the metrics of the Python files.
        conditions (dict[str, function]): Dictionary of conditions. Keys must match the header names of `metrics_file`. Values must be lambda functions.
        output_file (str): Filename of the .xlsx file that lists all filtered files that match all given conditions.

    Returns:
        pd.DataFrame: The dataframe containing the filtered rows from `metrics_file`. Represents the filtered Python scripts.
    """
    # Read metrics from .xlsx file
    df = pd.read_excel(metrics_file)
    
    # List to store filtered files
    filtered_files = []

    # Total number of rows in the dataframe
    total_files = len(df)
    
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        # Check that all conditions are met for each file
        if all(condition(row[column]) for column, condition in conditions.items() if column in row):
            filtered_files.append(row)

        # Print progress
        print(f"Processed {i}/{total_files} files", end='\r')
    
    # Filter columns to include only those referenced in the conditions
    condition_columns = list(conditions.keys())
    
    # Create dataframe for filtered files
    filtered_df = pd.DataFrame(filtered_files)
    
    # only select columns that exist in the DataFrame
    columns_to_save = ["file"] + [col for col in condition_columns if col in filtered_df.columns]
    
    # Save filtered files with only the relevant columns
    filtered_df = filtered_df[columns_to_save]
    filtered_df.to_excel(output_file, index=False)
    print(f" > Filtered {len(filtered_files)} out of {total_files} files and saved to {output_file}")

    return filtered_df

if __name__ == "__main__":

    # Define conditions for filtering
    conditions = {
        "lines": lambda x: 30 <= x <= 300, # between 30 and 300 lines
        "functions": lambda x: x >= 3, # at least three functions
        "functions_with_docstring": lambda x: x >= 3, # at least three functions with a docstring
        # add more conditions here if necessary ...
    }
    # Path to the .xlsx file containing the metrics
    metrics_file = "./script_metrics.xlsx"
    # Path to the .xlsx file in which to place the names of the filtered scripts
    output_file = "./filtered_files.xlsx"

    # Get the filtered files based on conditions
    filtered_df = filter_files_based_on_conditions(metrics_file, conditions, output_file)

    print(f"Done.")
