from os import PathLike

from hyperpyyaml import load_hyperpyyaml
from pandas import DataFrame, read_csv


def load_config(config_file: str | PathLike) -> dict:
    """
    Load the configuration from a YAML file and return it as a dictionary.

    Args:
        config_file (str | PathLike): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration data loaded from the YAML file.
    """
    with open(config_file, 'r') as file:
        return load_hyperpyyaml(file)


def records_to_csv(manifest_path: str | PathLike, content: list[dict]):
    """
    Export a list of dictionaries to a CSV file. Those are the subset manifests.

    Parameters:
        manifest_path (str or PathLike): The path where the CSV file will be saved.
        content (list of dict): The content to be exported, where each dictionary represents a row.

    Returns:
        None
    """
    manifest_data = DataFrame.from_records(content)
    manifest_data.to_csv(manifest_path, index=None)


def print_diaect_durations_in_manifest(csv_path: str | PathLike):
    """
    Print the duration and number of samples for each unique dialect in a dataset manifest.

    This function reads a dataset manifest CSV, calculates the total duration (in hours) for each dialect,
    and prints the number of samples and their respective durations. Additionally, it calculates and displays
    the overall total duration of all dialects in the dataset.

    Args:
        csv_path (str | PathLike): Path to the CSV file containing the dataset manifest. The manifest should
            have the following columns:
                - 'dialect': The dialect label for each sample.
                - 'duration': The duration of each sample in seconds.

    Returns:
        None
    """
    data = {}
    df = read_csv(csv_path)
    total_duration = 0
    for dial in list(df.dialect.unique()):
        dial_duration = sum(df[df['dialect'] == dial]['duration'] / 3600)
        total_duration += dial_duration

        data[dial] = dial_duration
        print(f"{dial}: {len(df[df['dialect'] == dial])} samples | duration: {round(dial_duration, 2)}")

    print(f"{round(total_duration, 2) = }")


def get_dialect_duration_from_records(records: list[dict], dialect: str) -> float:
    """
    Calculate the total duration for a specific dialect from a list of records.

    Args:
        records (list[dict]): A list of dictionaries representing dataset records.
                              Each record must contain 'dialect' and 'duration' keys.
        dialect (str): The target dialect to calculate the total duration for.

    Returns:
        float: The total duration (in seconds) of all records for the specified dialect.
               Returns 0.0 if no records match the given dialect or if the input is empty.

    Raises:
        ValueError: If the input records do not contain the required keys 'dialect' or 'duration'.
    """
    if not records:
        return 0.0

    df = DataFrame(records)

    if not {'dialect', 'duration'}.issubset(df.columns):
        raise ValueError("Each record must contain 'dialect' and 'duration' keys.")

    return df[df["dialect"] == dialect]["duration"].sum()


def remove_duplicates(starting_point_data: DataFrame | list[dict], data_pool: DataFrame) -> DataFrame:
    """
    Remove rows from the data pool that have duplicate 'ID' values already present in the starting point data.

    Args:
        starting_point_data (DataFrame | list[dict]): The starting dataset, either as a DataFrame or a list of records.
                                                      Each record or row must contain an 'ID' column.
        data_pool (DataFrame): The pool of additional data to filter, which must also contain an 'ID' column.

    Returns:
        DataFrame: A DataFrame containing only the rows from the data pool where 'ID' is not present in the starting point data.

    Raises:
        ValueError: If either input is missing the 'ID' column.
    """
    # Convert starting_point_data to DataFrame if it's a list
    if isinstance(starting_point_data, list):
        starting_point_data = DataFrame(starting_point_data)

    # Validate that both inputs contain the 'ID' column
    if 'ID' not in starting_point_data.columns:
        raise ValueError("The starting_point_data must contain an 'ID' column.")
    if 'ID' not in data_pool.columns:
        raise ValueError("The data_pool must contain an 'ID' column.")

    # Filter data_pool by excluding rows with 'ID' values present in starting_point_data
    filtered_data = data_pool[~data_pool['ID'].isin(starting_point_data['ID'])]

    return filtered_data
