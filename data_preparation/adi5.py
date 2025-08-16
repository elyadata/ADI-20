"""
Script to compile statistics for the ADI-5 dataset and create a manifest file.

This script processes audio data by extracting statistics from WAV files in the dataset,
compiles global statistics across subsets, and generates a CSV manifest with the audio segment details.

Usage:
    python script_name.py --dataset_path /path/to/dataset --stats_path /path/to/stats --manifest_path /path/to/manifest
    or import the main function in your script

Arguments:
    --dataset_path   : Path to the dataset containing the audio files.
    --stats_path     : Directory to save the statistics output (default: current directory)
    --manifest_path  : Directory to save the manifest CSV file (default: current directory)

Author: Haroun Elleuch, 2024
"""

import wave
from argparse import ArgumentParser
from contextlib import closing
from dataclasses import dataclass, field
from glob import glob
from os import PathLike
from os import makedirs
from os.path import join

from loguru import logger
from numpy import array, ndarray, mean, append, average
from numpy import sum as np_sum
from tqdm import tqdm

from utils import records_to_csv

ADI_SAMPLE_RATE = 16_000


@dataclass
class DataStats:
    """
    A data class representing statistics for a collection of audio data segments.

    Attributes:
        name (str): The name of the dataset or subset.
        number_of_segments (int): The total number of audio segments.
        total_duration (float): The total duration of all segments, in seconds.
        avg_duration (float): The average duration of the audio segments, in seconds.
        min_duration (float): The minimum duration of the audio segments, in seconds.
        max_duration (float): The maximum duration of the audio segments, in seconds.
        durations (ndarray): An array of individual segment durations, in seconds.
    """

    name: str = "ADI-5"
    number_of_segments: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    durations: ndarray = field(default_factory=lambda: array([]))

    def __add__(self, other):
        """
        Add two DataStats objects.

        Args:
            other (DataStats): Another DataStats object to add.

        Returns:
            DataStats: A new DataStats object with combined statistics.
        """
        combined_durations = append(self.durations, other.durations)
        return DataStats(
            number_of_segments=self.number_of_segments + other.number_of_segments,
            total_duration=self.total_duration + other.total_duration,
            avg_duration=mean(combined_durations),
            min_duration=combined_durations.min(),
            max_duration=combined_durations.max(),
            durations=combined_durations
        )

    def __str__(self):
        """
        Return a string representation of the DataStats object.

        Returns:
            str: A string representation of the DataStats object.
        """
        return (f"DataStats({self.name}: "
                f"number_of_segments={self.number_of_segments}, "
                f"total_duration={self.total_duration:.2f}, "
                f"avg_duration={self.avg_duration:.2f}, "
                f"min_duration={self.min_duration:.2f}, "
                f"max_duration={self.max_duration:.2f}")


def get_wav_duration(file_path: str | PathLike) -> float:
    """
    Get the duration of a WAV file.

    Args:
        file_path (str | PathLike): The path to the WAV file.
    Returns:
       duration (float): Duration of the audio in seconds.

    """
    with closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def get_subset_stats(subset_path: str | PathLike) -> tuple[DataStats, list[dict]]:
    """
    Calculate statistics for a subset of audio data and generate a manifest.

    Args:
        subset_path (str | PathLike): The directory path to the subset containing WAV files.

    Returns:
        tuple[DataStats, list[dict]]: A tuple containing:
            - DataStats: An object holding the statistics for the subset.
            - list[dict]: A manifest list, where each entry contains details about a WAV file
              (e.g., path, dialect, duration).
    """

    durations = []
    manifest_data = []
    dialect = subset_path.split("/")[-2]

    print(f"{dialect = }")
    print(f"{subset_path = }")

    subset_wav_files = glob(f"{subset_path}/**/*.wav", recursive=True)

    progress_bar_msg = f"Processing wav files in {subset_path.split('/')[-1]}"

    for wav in tqdm(subset_wav_files, desc=progress_bar_msg, unit_scale=True):
        duration = get_wav_duration(wav)
        durations.append(duration)
        manifest_data.append(
            {
                "ID": wav.split("/")[-1].replace(".wav", ""),
                "wav": wav,
                "dialect": dialect,
                "duration": get_wav_duration(wav),
                "start": 0,
                "end": int(duration * ADI_SAMPLE_RATE),
            }
        )

    durations = array(durations)

    return DataStats(
        name=dialect,
        number_of_segments=len(subset_wav_files),
        total_duration=np_sum(durations),
        avg_duration=average(durations),
        min_duration=durations.min(),
        max_duration=durations.max(),
        durations=durations
    ), manifest_data


def save_stats(output_path: str | PathLike, global_stats: DataStats,
               subset_stats: list[DataStats], split: str | None) -> None:
    """
    Save the computed statistics to a text file.

    Args:
        output_path (str | PathLike): Directory where the statistics file will be saved.
        global_stats (DataStats): The overall statistics of the entire dataset.
        subset_stats (list[DataStats]): A list of DataStats objects for individual subsets.
    """
    if split:
        output_file = f"{split}_stats.txt"
    else:
        output_file = "stats.txt"

    with open(join(output_path, output_file), mode="w") as f:
        f.write("Global stats: \n")
        f.write(str(global_stats))
        f.write("\n*****" * 10)

        for stats in subset_stats:
            f.write(f"\n{str(stats)}")
            f.write("_____" * 7)


def compile_dataset_stats(dataset_path: str | PathLike,
                          split: str,
                          stats_path: str | PathLike,
                          manifest_path: str | PathLike):
    """
    Compile and save statistics and manifest for the entire dataset.

    Args:
        dataset_path (str | PathLike): The path to the dataset's root directory containing audio files.
        split (str): Split of the dataste. Either train, dev or test.
        stats_path (str | PathLike): The directory where the computed statistics will be saved.
        manifest_path (str | PathLike): The directory where the manifest CSV will be saved.
    """
    train_dirs = glob(f"{dataset_path}/{split}/wav/**/")
    global_stats = DataStats()
    subsets_stats = []
    subsets_manifests = []

    for dir_ in tqdm(train_dirs, desc="processing sub-directories:"):
        dir_stats, sub_manifest = get_subset_stats(dir_)
        global_stats += dir_stats
        subsets_stats.append(dir_stats)
        subsets_manifests.extend(sub_manifest)
        logger.info(f"{dir_stats}")

    logger.info(f"Global stats: {global_stats}")
    save_stats(output_path=stats_path, split=split, global_stats=global_stats, subset_stats=subsets_stats)
    records_to_csv(manifest_path=join(manifest_path, f"adi5_{split}.csv"), content=subsets_manifests)


def main(dataset_path: str | PathLike,
         stats_path: str | PathLike,
         manifest_path: str | PathLike) -> None:
    """
    Main function to process the ADI 17 dataset, compute statistics, and generate manifest files
    for evaluation splits. It handles both full dataset processing (statistics generation and
    manifest creation) or evaluation-only processing based on the `eval_only` flag.

    Args:
        dataset_path (str | PathLike): Path to the dataset directory containing audio files and labels.
        split (str): Split of the dataste. Either train, dev or test.
        stats_path (str | PathLike): Directory where the statistics file will be saved.
        manifest_path (str | PathLike): Directory where the manifest CSV files for the evaluation
                                        splits will be saved.

    Returns:
        None: The function does not return a value but generates statistics and manifest files.
    """

    logger.info("Compiling ADI 5 stats...")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Statistics path: {args.stats_path}")
    logger.info(f"Manifest path: {args.manifest_path}")

    makedirs(stats_path, exist_ok=True)
    makedirs(manifest_path, exist_ok=True)

    for split in ["train", "test", "dev"]:
        logger.info(f"Generating {split} manifest...")
        compile_dataset_stats(
            dataset_path=dataset_path,
            split=split,
            stats_path=stats_path,
            manifest_path=manifest_path)

    logger.info("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Process some paths.")
    parser.add_argument('--dataset_path', type=str,
                        default='/media/elya1/corpus/mgb3/mgb3-ADI5/', help='Path to the dataset')
    parser.add_argument('--stats_path', type=str, default='adi5_manifests', help='Path to save the stats file')
    parser.add_argument('--manifest_path', type=str, default='adi5_manifests', help='Path to save the manifest file')

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        manifest_path=args.manifest_path,
        stats_path=args.stats_path,
    )
