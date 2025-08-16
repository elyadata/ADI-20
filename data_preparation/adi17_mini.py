#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to create a mini version of the ADI-17 dataset based on the provided configuration file.

Usage:
    python script_name.py --config_file /path/to/config_file.yaml
    Or import the main function in your code.

Arguments:
    --config_file : Path to the YAML configuration file containing:
        - manifest_paths: Paths to the original dataset and the mini dataset output.
        - filters: Parameters for filtering (segment duration, deviation, hours per dialect).
        - seed: Seed for random sampling.

Author: Haroun Elleuch, 2024
"""

import random
from argparse import ArgumentParser
from os import makedirs, PathLike

import pandas as pd

from utils import load_config, records_to_csv


def sample_dataset(manifest_data_frame: pd.DataFrame, seed: int, hours_per_dialect: float) -> list[dict]:
    """
    Sample a specified number of hours from each dialect subset of ADI-17 and return a compiled list of records.

    Args:
        manifest_data_frame (pd.DataFrame): A DataFrame containing the dataset, with columns such as 'dialect' and 'duration'.
        seed (int): A random seed to ensure reproducibility of the sampling process.
        hours_per_dialect (float): The number of hours to sample from each dialect subset, which will be converted to seconds.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary corresponds to a sampled record from the DataFrame.
                    Each dictionary contains details of a single data entry (e.g., 'ID', 'wav', 'dialect', 'duration').

    Process:
        - The function groups the data by 'dialect'.
        - For each dialect group, the records are shuffled (using the provided seed for reproducibility).
        - The function iterates through each group and adds records until the cumulative duration of sampled records
          reaches the specified number of hours (converted to seconds).
        - Once the target duration for a dialect is reached, the sampling stops, and the process continues for the next dialect.
    """
    random.seed(seed)

    # List to store the sampled records
    sampled_records = []

    # Target duration in seconds
    target_duration = hours_per_dialect * 3600  # Convert hours to seconds

    # Group dataset by dialect
    for dialect, group in manifest_data_frame.groupby('dialect'):
        # Shuffle and reset index for reproducibility
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Initialize cumulative duration
        cumulative_duration = 0
        sampled_group = []

        # Iterate through rows and accumulate duration until target is reached
        for _, row in group.iterrows():
            if cumulative_duration + row['duration'] <= target_duration:
                sampled_group.append(row.to_dict())
                cumulative_duration += row['duration']
            else:
                break

        # Extend sampled records
        sampled_records.extend(sampled_group)

    return sampled_records


def filter_dataset(manifest_data_frame: pd.DataFrame, dialects: list[str], filters: dict) -> pd.DataFrame:
    """
    Filter the ADI-17 dataset by the provided dialects and duration constraints.

    Args:
        manifest_data_frame (pd.DataFrame): A DataFrame containing the dataset with columns such as 'dialect' and 'duration'.
        dialects (list[str]): A list of dialects to retain in the filtered dataset.
        filters (dict): A dictionary containing the following keys:
            - "target_segment_duration" (float): The desired segment duration in seconds.
            - "segment_duration_deviation" (float): The allowable deviation from the target segment duration.

    Returns:
        pd.DataFrame: A DataFrame containing only the records that match the specified dialects and whose 'duration'
                      falls within the target range (i.e., between `target_segment_duration` ± `segment_duration_deviation`).

    Process:
        - First, the function filters the dataset to keep only rows whose 'dialect' matches the specified list of dialects.
        - Then, it calculates the minimum and maximum acceptable segment durations based on the target duration and deviation.
        - Finally, it filters the dataset again to keep only records whose 'duration' falls within the specified bounds.
    """
    min_segment_length = filters["target_segment_duration"] - filters["segment_duration_deviation"]
    max_segment_length = filters["target_segment_duration"] + filters["segment_duration_deviation"]

    # Filter based on dialects
    filtered_df = manifest_data_frame[manifest_data_frame['dialect'].isin(dialects)]

    # Filter based on duration within bounds
    filtered_df = filtered_df[
        (filtered_df['duration'] >= min_segment_length) &
        (filtered_df['duration'] <= max_segment_length)
    ]

    return filtered_df


def get_dialects(manifest_data_frame: pd.DataFrame) -> list[str]:
    """
    Get the list of unique dialects from the manifest DataFrame.

    Args:
        manifest_data_frame (pd.DataFrame): A DataFrame containing the dataset with a 'dialect' column.

    Returns:
        list[str]: A list of unique dialects present in the 'dialect' column of the DataFrame.

    Process:
        - The function extracts the unique values from the 'dialect' column.
        - It converts these unique values into a list and returns it.
    """
    return manifest_data_frame['dialect'].unique().tolist()


def create_adi17_mini(config: dict) -> list[dict]:
    """
    Create a mini version of the ADI-17 dataset based on the provided config settings.

    Args:
        config (dict): A configuration dictionary containing:
            - "manifest_paths" (dict): A dictionary with paths to the dataset manifest, including:
                - "adi17" (str): Path to the ADI-17 dataset CSV file.
            - "filters" (dict): A dictionary containing filters to apply, such as:
                - "target_segment_duration" (float): The desired segment duration in seconds.
                - "segment_duration_deviation" (float): Allowed deviation from the target segment duration.
                - "hours_per_dialect" (float): Number of hours to sample per dialect.
            - "seed" (int): A random seed for reproducibility when sampling the dataset.

    Returns:
        list[dict]: A list of dictionaries representing the mini version of the ADI-17 dataset, with sampled records
                    for each dialect based on the provided filters and sampling strategy.

    Process:
        1. The dataset is loaded from the CSV file specified in the `config["manifest_paths"]["adi17"]`.
        2. The unique dialects in the dataset are extracted.
        3. The dataset is filtered based on the provided dialects and duration constraints in the `config["filters"]`.
        4. A specified number of hours of data is sampled for each dialect, and the resulting records are compiled.
        5. The final list of sampled records is returned.
    """
    # Load the dataset from the manifest path
    manifest: pd.DataFrame = pd.read_csv(config["manifest_paths"]["adi17_segmented"])

    # Get unique dialects from the dataset
    dialects: list[str] = get_dialects(manifest)

    # Filter the dataset by the provided filters
    filtered_dataset: pd.DataFrame = filter_dataset(
        manifest_data_frame=manifest,
        dialects=dialects,
        filters=config["filters"]
    )

    # Sample records for each dialect
    adi_mini_data: list[dict] = sample_dataset(
        manifest_data_frame=filtered_dataset,
        hours_per_dialect=config["filters"]["hours_per_dialect"],
        seed=config["seed"]
    )

    return adi_mini_data


def main(config_file: str | PathLike):
    """
    Main function to create a mini version of the ADI-17 dataset based on the configuration file.

    Args:
        config_file (str | PathLike): Path to the YAML configuration file that contains the paths, filters,
                                      and other parameters required to create the mini dataset.

    Process:
        1. Load the configuration from the provided YAML file using `load_config()`.
        2. Ensure the output directory for the mini dataset exists, creating it if necessary.
        3. Generate a mini version of the ADI-17 dataset by filtering and sampling records according to the config settings.
        4. Write the resulting mini dataset to a CSV file in the specified output directory.

    Configuration Requirements:
        The `config_file` should define:
            - manifest_paths (dict): Paths for the original and mini dataset manifests.
            - filters (dict): Filtering parameters, including target segment duration, deviation, and hours per dialect.
            - seed (int): Seed value for reproducible random sampling.

    Output:
        - The mini dataset is saved as a CSV file at the specified location.
    """

    # Load configuration file
    config = load_config(config_file)

    # Ensure the output directory exists
    output_dir = config["manifest_paths"].get("adi17_mini", "adi17_mini.csv")
    filename = output_dir.split("/")[-1]
    makedirs(output_dir.replace(filename, ""), exist_ok=True)

    # Generate the mini dataset
    adi17_mini_data = create_adi17_mini(config)

    # Write the mini dataset to CSV
    records_to_csv(
        manifest_path=output_dir,
        content=adi17_mini_data,
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Create a mini version of the ADI-17 dataset")
    arg_parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    args = arg_parser.parse_args()

    main(args.config_file)
