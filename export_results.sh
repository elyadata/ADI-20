#!/bin/bash

# Base directory containing the results
RESULTS_DIR="model/results"
OUTPUT_CSV="experiment_results.csv"

# Check if the results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Directory $RESULTS_DIR does not exist."
    exit 1
fi

# Write the CSV header
echo "Experiment,Last Epoch,Last Weighted F1,Best Epoch,Best Weighted F1,Best Weighted Precision,Best Weighted Recall" > "$OUTPUT_CSV"

# Use find to locate all train_log.txt files and collect relevant information
find "$RESULTS_DIR" -name "train_log.txt" | while read -r LOG_FILE; do
    # Get the experiment directory including the seed and ignoring "results"
    EXPERIMENT_DIR=$(dirname "$LOG_FILE" | awk -F '/' '{if ($(NF-2) != "results") {print $(NF-2) "/" $(NF-1) "/" $NF}else{print $(NF-1) "/" $NF}}' | sed 's|/[0-9]*$||')

    # Initialize variables for last epoch and F1
    LAST_EPOCH=""
    LAST_WEIGHTED_F1=""
    BEST_EPOCH=""
    BEST_F1=0
    BEST_PRECISION=0
    BEST_RECALL=0

    # Read the log file line by line
    while read -r LINE; do
        # Extract epoch number
        if [[ "$LINE" =~ Epoch:\ ([0-9]+) ]]; then
            EPOCH="${BASH_REMATCH[1]}"
        fi

        # Extract weighted F1 score, precision, and recall
        if [[ "$LINE" =~ valid\ weighted_f1:\ ([0-9]+\.[0-9]+) ]]; then
            WEIGHTED_F1="${BASH_REMATCH[1]}"
            WEIGHTED_F1=$(echo "$WEIGHTED_F1 * 10" | bc) # Scale for display
        fi
        if [[ "$LINE" =~ valid\ weighted_precision:\ ([0-9]+\.[0-9]+) ]]; then
            WEIGHTED_PRECISION="${BASH_REMATCH[1]}"
            WEIGHTED_PRECISION=$(echo "$WEIGHTED_PRECISION * 10" | bc) # Scale for display
        fi
        if [[ "$LINE" =~ valid\ weighted_recall:\ ([0-9]+\.[0-9]+) ]]; then
            WEIGHTED_RECALL="${BASH_REMATCH[1]}"
            WEIGHTED_RECALL=$(echo "$WEIGHTED_RECALL * 10" | bc) # Scale for display
        fi

        # Update the last epoch and last F1 score
        if [ -n "$EPOCH" ] && [ -n "$WEIGHTED_F1" ]; then
            LAST_EPOCH="$EPOCH"
            LAST_WEIGHTED_F1="$WEIGHTED_F1"
        fi

        # Check if this F1 score is the best
        if [ -n "$WEIGHTED_F1" ] && (( $(echo "$WEIGHTED_F1 > $BEST_F1" | bc -l) )); then
            BEST_F1="$WEIGHTED_F1"
            BEST_EPOCH="$EPOCH"
            BEST_PRECISION="$WEIGHTED_PRECISION"
            BEST_RECALL="$WEIGHTED_RECALL"
        fi
    done < "$LOG_FILE"

    # Write the results to the CSV file
    if [ -n "$LAST_EPOCH" ] && [ -n "$LAST_WEIGHTED_F1" ] && [ -n "$BEST_EPOCH" ] && [ -n "$BEST_F1" ]; then
        echo "$EXPERIMENT_DIR,$LAST_EPOCH,$LAST_WEIGHTED_F1,$BEST_EPOCH,$BEST_F1,$BEST_PRECISION,$BEST_RECALL" >> "$OUTPUT_CSV"
    fi
done

# If no train_log.txt files were found
if [ $? -ne 0 ]; then
    echo "No train_log.txt files found in $RESULTS_DIR."
fi

# Notify the user of the output location
echo "Results saved to $OUTPUT_CSV"
