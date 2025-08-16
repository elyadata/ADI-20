#!/usr/bin/env python3
"""
Script to split long audio segments from an ADI-17 dataset and save an updated manifest.

This script processes a CSV manifest by splitting segments longer than a specified duration into smaller chunks.
Only fields related to `ID`, `start`, `end`, and `duration` are updated; other fields are duplicated across splits.

Example command:
    python split_manifest.py --input_csv adi17.csv --output_csv adi17_prepared.csv --max_segment_duration 30 --desired_split_duration 12
"""

import argparse
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths and parameters.
    """
    parser = argparse.ArgumentParser(description="Split long audio segments and update the ADI-17 manifest.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV manifest file.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV manifest file.")
    parser.add_argument('--max_segment_duration', type=float, default=30.0,
                        help="Maximum segment duration (seconds) before splitting. Default is 30 seconds.")
    parser.add_argument('--desired_split_duration', type=float, default=12.0,
                        help="Desired split duration (seconds) for long segments. Default is 10 seconds.")
    parser.add_argument('--sampling_rate', type=int, default=16000,
                        help="Sampling rate of the audio files in Hz. Default is 16kHz.")

    return parser.parse_args()


def split_segment(row: pd.Series, desired_split_duration: float, sampling_rate: int) -> list[dict]:
    """
    Split a long audio segment into smaller chunks of the desired duration.

    Args:
        row (pd.Series): A row of the DataFrame representing a segment.
        desired_split_duration (float): The desired duration of the split segments in seconds.
        sampling_rate (int): The sampling rate of the audio files in Hz.

    Returns:
        list[dict]: A list of dictionaries, each representing a split segment.
    """
    splits = []
    duration = row['duration']
    start = row['start']
    end = row['end']
    segment_id = row['ID']

    num_splits = int(np.ceil(duration / desired_split_duration))

    for i in range(num_splits):
        new_start = start + i * desired_split_duration * sampling_rate
        new_end = min(start + (i + 1) * desired_split_duration * sampling_rate, end)
        new_duration = (new_end - new_start) / sampling_rate

        new_segment = row.to_dict()
        new_segment['ID'] = f"{segment_id}_split_{i + 1}"
        new_segment['start'] = new_start
        new_segment['end'] = new_end
        new_segment['duration'] = new_duration

        splits.append(new_segment)

    return splits


def get_manifest_differences(original_df: pd.DataFrame, updated_df: pd.DataFrame) -> None:
    """
    Print the differences in the manifest before and after processing.

    Args:
        original_df (pd.DataFrame): The original DataFrame before processing.
        updated_df (pd.DataFrame): The updated DataFrame after processing.
    """
    original_segment_count = len(original_df)
    updated_segment_count = len(updated_df)

    original_average_duration = original_df['duration'].mean()
    updated_average_duration = updated_df['duration'].mean()

    original_total_duration = original_df['duration'].sum()
    updated_total_duration = updated_df['duration'].sum()

    logger.info("Manifest Differences:")
    logger.info(f"Number of segments - Original: {original_segment_count}, Updated: {updated_segment_count}")
    logger.info(
        f"Average segment duration (seconds) - Original: {original_average_duration:.2f}, Updated: {updated_average_duration:.2f}")
    logger.info(
        f"Total duration (seconds) - Original: {original_total_duration:.2f}, Updated: {updated_total_duration:.2f}")


def process_manifest(input_csv: str,
                     output_csv: str,
                     max_segment_duration: float = 30.0,
                     desired_split_duration: float = 12.0,
                     sampling_rate: int = 16_000) -> None:
    """
    Process the CSV manifest to split long audio segments and save the updated manifest.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
        max_segment_duration (float): Maximum duration for segments before splitting.
        desired_split_duration (float): Desired duration for split segments.
        sampling_rate (int): Sampling rate of the audio files.
    """
    df = pd.read_csv(input_csv)

    long_segments = df[df['duration'] > max_segment_duration]

    tqdm.pandas(desc="Processing segments")

    split_segments = long_segments.progress_apply(
        split_segment,
        axis=1,
        desired_split_duration=desired_split_duration,
        sampling_rate=sampling_rate
    )

    split_segments_flat = [segment for sublist in split_segments for segment in sublist]
    split_df = pd.DataFrame(split_segments_flat)

    df_filtered = df[df['duration'] <= max_segment_duration]
    df_final = pd.concat([df_filtered, split_df], ignore_index=True)

    df_final.to_csv(output_csv, index=False)

    logger.info(f"Updated manifest saved to {output_csv}")
    get_manifest_differences(original_df=df, updated_df=df_final)


def main() -> None:
    """
    Main function to execute the script as a standalone program.
    """
    args = parse_args()
    process_manifest(
        args.input_csv,
        args.output_csv,
        args.max_segment_duration,
        args.desired_split_duration,
        args.sampling_rate)


if __name__ == "__main__":
    main()
