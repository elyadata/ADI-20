"""
Script to create an augmented version of the ADI-17 dataset based on the provided configuration file.
This enables the incorporation of multiple dataframes for external data.
This will output an ADI-20 dataset and a 53h per dialect version of ADI-17 (medium) based on curated data.
Usage:
    python script_name.py --config_file /path/to/config_file.yaml
    Or import the main function in your code.

Arguments:
    config_file : Path to the YAML configuration file.

Author: Haroun Elleuch, 2024
"""

import logging
import random
from argparse import ArgumentParser
from os import PathLike, makedirs, path

import pandas as pd
from tqdm import tqdm
from utils import (get_dialect_duration_from_records, load_config,
                   print_diaect_durations_in_manifest, records_to_csv, remove_duplicates)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sample_dataset(manifest_data_frame: pd.DataFrame, seed: int, hours_per_dialect: float,
                   starter_data_records: list[dict] | None = None) -> list[dict]:
    """
    Sample a specified number of hours from each dialect subset of ADI-17 and return a compiled list of records.

    Args:
        manifest_data_frame (pd.DataFrame): A DataFrame containing the dataset, with columns such as 'dialect' and 'duration'.
        seed (int): A random seed to ensure reproducibility of the sampling process.
        hours_per_dialect (float): The number of hours to sample from each dialect subset, which will be converted to seconds.
        starter_data_records (list[dict] | None): Optional starting data to include in the sampling.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary corresponds to a sampled record from the DataFrame.
    """
    random.seed(seed)

    # Input validation
    if not {'dialect', 'duration'}.issubset(manifest_data_frame.columns):
        raise ValueError("Input DataFrame must contain 'dialect' and 'duration' columns.")
    if hours_per_dialect <= 0:
        raise ValueError("hours_per_dialect must be a positive number.")

    # Deduplicate input DataFrame
    manifest_data_frame = manifest_data_frame.drop_duplicates(subset='ID')

    # Initialize sampled records
    if starter_data_records is None:
        sampled_records = []
    else:
        manifest_data_frame = remove_duplicates(starting_point_data=starter_data_records, data_pool=manifest_data_frame)
        if isinstance(starter_data_records, pd.DataFrame):
            sampled_records = starter_data_records.to_dict(orient="records")
        elif isinstance(starter_data_records, list):
            sampled_records = starter_data_records
        else:
            raise TypeError("Invalid type for starter_data_records. Should be either list or pd.DataFrame.")

    target_duration = hours_per_dialect * 3600  # Convert hours to seconds

    # Group dataset by dialect
    for dialect, group in tqdm(manifest_data_frame.groupby('dialect'), desc="Sampling dialects..."):
        # Shuffle and reset index for reproducibility
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Initialize cumulative duration
        cumulative_duration = get_dialect_duration_from_records(records=sampled_records, dialect=dialect)
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


def make_dataset_from_starting_point(
    data_pool: list[pd.DataFrame],
    starting_point: pd.DataFrame | None | list[dict],
    target_dialect_duration: int | None
):
    """
    Creates an ADI dataset from a starting point.

    Parameters
    ----------
    starting_point : pd.DataFrame
        Starting point of the dataset creation.
    data_pool : list[pd.DataFrame]
        A list of additional datasets to be either added or sampled from.
    target_dialect_duration : int | None
        Desired duration per dialect. If None, all the data will be concatenated.

    Returns
    -------
    list[dict]: The resulting dataset as a list of records.
    """

    # Combine datasets and deduplicate
    combined_data_pool = pd.concat(data_pool, ignore_index=True).drop_duplicates(subset='ID')

    if target_dialect_duration is None:
        if starting_point is not None:
            combined_data_pool = pd.concat([combined_data_pool, starting_point],
                                           ignore_index=True).drop_duplicates(subset='ID')
        logging.info(f"Created dataset with {len(combined_data_pool)} records (no sampling).")
        return combined_data_pool.to_dict(orient='records')

    elif target_dialect_duration > 0:
        # Sample the combined dataset
        sampled_data = sample_dataset(
            manifest_data_frame=combined_data_pool,
            starter_data_records=starting_point,
            seed=42,
            hours_per_dialect=target_dialect_duration
        )
        logging.info(
            f"Created sampled dataset with {len(sampled_data)} records (target duration: {target_dialect_duration}h per dialect).")
        return sampled_data
    else:
        raise ValueError(f"Invalid value {target_dialect_duration} for target_dialect_duration provided.")


def main(config_file: str | PathLike):
    logging.info("Setting up...")
    config = load_config(config_file)

    # Create the directories if they don't exist
    for folder in set(path.dirname(p) for p in config["output_manifest_paths"].values()):
        makedirs(folder, exist_ok=True)

    # Read input data
    logging.info("Reading input data...")
    adi17 = pd.read_csv(config["input_manifest_paths"]["adi17"])
    adi17_medium = pd.read_csv(config["input_manifest_paths"]["adi17_medium"])
    jordanian = pd.read_csv(config["input_manifest_paths"]["jordanian"])
    sudanese = pd.read_csv(config["input_manifest_paths"]["sudanese"])
    tunisian = pd.read_csv(config["input_manifest_paths"]["tunisian"])
    msa = pd.read_csv(config["input_manifest_paths"]["msa"])
    bahraini = pd.read_csv(config["input_manifest_paths"]["bahraini"])

    # Create ADI-20-medium
    logging.info("Creating ADI-20-medium dataset...")
    adi20_medium_data = make_dataset_from_starting_point(
        starting_point=adi17_medium,
        data_pool=[tunisian, msa, bahraini],
        target_dialect_duration=53
    )
    records_to_csv(manifest_path=config["output_manifest_paths"]["adi20_medium"], content=adi20_medium_data)
    logging.info("ADI-20-medium sampled. Stats:")
    print_diaect_durations_in_manifest(config["output_manifest_paths"]["adi20_medium"])

    # Create ADI-20
    logging.info("Creating the complete version of ADI-20 dataset...")
    adi20_data = make_dataset_from_starting_point(
        starting_point=adi17,
        data_pool=[jordanian, sudanese, tunisian, msa, bahraini],
        target_dialect_duration=None
    )
    records_to_csv(manifest_path=config["output_manifest_paths"]["adi20"], content=adi20_data)
    logging.info("ADI-20 compiled. Stats:")
    print_diaect_durations_in_manifest(config["output_manifest_paths"]["adi20"])

    logging.info("Creating ADI-20 dev manifest...")
    dev_adi17 = pd.read_csv(config["dev_manifests"]["adi17"])
    dev_tn = pd.read_csv(config["dev_manifests"]["tunisian"])
    dev_msa = pd.read_csv(config["dev_manifests"]["msa"])
    dev_msa = dev_msa[dev_msa["dialect"] == "MSA"]
    dev_bahraini = pd.read_csv(config["dev_manifests"]["bahraini"])

    pd.concat([dev_adi17, dev_tn, dev_msa, dev_bahraini], ignore_index=True).to_csv(
        config["output_manifest_paths"]["adi20_dev"], index=False)
    logging.info("Stats of ADI-20 dev set:")
    print_diaect_durations_in_manifest(config["output_manifest_paths"]["adi20_dev"])

    logging.info("Creating ADI-20 test manifest...")
    test_adi17 = pd.read_csv(config["test_manifests"]["adi17"])
    test_tn = pd.read_csv(config["test_manifests"]["tunisian"])
    test_msa = pd.read_csv(config["test_manifests"]["msa"])
    test_msa = test_msa[test_msa["dialect"] == "MSA"]
    test_bahraini = pd.read_csv(config["test_manifests"]["bahraini"])

    pd.concat([test_adi17, test_tn, test_msa, test_bahraini], ignore_index=True).to_csv(
        config["output_manifest_paths"]["adi20_test"], index=False)
    logging.info("Sanity check for ADI-20 test set:")
    print_diaect_durations_in_manifest(config["output_manifest_paths"]["adi20_test"])

    logging.info("Done.")


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="Create the new ADI-20 dataset \
        and its medium version with added Tunisian and Bahraini dialects and MSA."
    )
    arg_parser.add_argument("config_file", type=str, help="Path to the config file")
    args = arg_parser.parse_args()

    main(args.config_file)
