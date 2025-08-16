#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to create an augmented version of the ADI-17 dataset based on the provided configuration file.
This enables the incorporation of multiple dataframes for external data.

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

from argparse import ArgumentParser
from os import PathLike, makedirs

import pandas as pd
from utils import load_config, records_to_csv

from adi17_mini import (filter_dataset, get_dialects,
                                         sample_dataset)


def create_adi17_medium(config: dict) -> list[dict]:
    """
    Create a mini version of the ADI-17 dataset based on the provided config settings.

    Args:
        config (dict): configuration dict.

    Returns:
        list[dict]: A list of dictionaries representing the medium version of the ADI-17 dataset, with sampled records
                    for each dialect based on the provided filters and sampling strategy.
    """
    # Load the dataset from the manifest path
    adi17: pd.DataFrame = pd.read_csv(config["manifest_paths"]["adi17_segmented"])
    jordanian: pd.DataFrame = pd.read_csv(config["manifest_paths"]["jordanian_data"])
    sudanese: pd.DataFrame = pd.read_csv(config["manifest_paths"]["sudanese_data"])

    sudanese["wav"] = config["data_paths"]["sudanese_data"] + \
        sudanese["wav"].str.replace("\\", "/")  # already done in the manifest

    manifest = pd.concat([adi17, jordanian, sudanese], ignore_index=True)

    # Get unique dialects from the dataset
    dialects: list[str] = get_dialects(manifest)

    # Filter the dataset by the provided filters
    filtered_dataset: pd.DataFrame = filter_dataset(
        manifest_data_frame=manifest,
        dialects=dialects,
        filters=config["filters"]
    )

    # Sample records for each dialect
    adi_medium_data: list[dict] = sample_dataset(
        manifest_data_frame=filtered_dataset,
        hours_per_dialect=config["filters"]["hours_per_dialect"],
        seed=config["seed"]
    )

    return adi_medium_data


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
    output_dir = config["manifest_paths"].get("adi17_medium", "adi17_medium_40h.csv")
    filename = output_dir.split("/")[-1]
    makedirs(output_dir.replace(filename, ""), exist_ok=True)

    # Generate the mini dataset
    adi17_medium_data = create_adi17_medium(config)

    # Write the mini dataset to CSV
    records_to_csv(
        manifest_path=output_dir,
        content=adi17_medium_data,
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Create a 40h (medium) version of the ADI-17 dataset")
    arg_parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    args = arg_parser.parse_args()

    main(args.config_file)
