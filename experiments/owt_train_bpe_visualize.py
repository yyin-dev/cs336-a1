import re
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import statistics

"""
BPE training log file format: 

2025-07-15 23:09:09,441 - INFO - vocab len: 257
Selected pair: (b' ', b't')
2025-07-15 23:09:12,059 - INFO - vocab len: 258
Selected pair: (b' ', b'a')
2025-07-15 23:09:14,752 - INFO - vocab len: 259
Selected pair: (b'h', b'e')
2025-07-15 23:09:21,144 - INFO - vocab len: 260
Selected pair: (b'i', b'n')
"""

def parse_log_file(filename):
    """
    Parse log file and extract timestamps, vocab lengths, and selected pairs.

    Args:
        filename (str): Path to the log file

    Returns:
        tuple: (timestamps, lengths, pairs) - lists of parsed data
    """
    timestamps = []
    lengths = []
    pairs = []

    # Regular expressions
    vocab_pattern = (
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - vocab len: (\d+)"
    )
    pair_pattern = r"Selected pair: (.+)"

    try:
        with open(filename, "r") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                # Check for vocab len line
                vocab_match = re.match(vocab_pattern, line)
                if vocab_match:
                    timestamp_str = vocab_match.group(1)
                    length = int(vocab_match.group(2))

                    # Parse timestamp
                    try:
                        timestamp = datetime.strptime(
                            timestamp_str, "%Y-%m-%d %H:%M:%S,%f"
                        )
                        timestamps.append(timestamp)
                        lengths.append(length)
                        pairs.append(None)  # No pair for vocab len lines
                    except ValueError as e:
                        print(
                            f"Warning: Could not parse timestamp on line {line_num}: {timestamp_str}"
                        )
                        continue

                # Check for selected pair line
                pair_match = re.search(pair_pattern, line)
                if pair_match:
                    pair_info = pair_match.group(1)
                    pairs.append(pair_info)
                    timestamps.append(None)  # No timestamp for pair lines
                    lengths.append(None)  # No length for pair lines

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return [], [], []
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], [], []

    return timestamps, lengths, pairs


def analyze_processing_times(timestamps, lengths, pairs):
    """
    Analyze processing times for vocab pairs and print statistics.

    Args:
        timestamps (list): List of datetime objects and None values
        lengths (list): List of vocab lengths and None values
        pairs (list): List of pair info and None values
    """
    # Find processing times for each pair
    processing_times = []
    slow_pairs: list[tuple[str, float]] = []  # Pairs that take more than 1 minute

    i = 0
    while i < len(timestamps):
        # Look for pattern: timestamp1 (vocab len) -> pair -> timestamp2 (vocab len)
        if (
            timestamps[i] is not None
            and i + 2 < len(timestamps)
            and pairs[i + 1] is not None
            and timestamps[i + 2] is not None
        ):

            start_time = timestamps[i]
            pair_info = pairs[i + 1]
            end_time = timestamps[i + 2]

            processing_time = (end_time - start_time).total_seconds()
            processing_times.append(processing_time)

            # Check if processing time > 1 minute
            if processing_time > 60:
                slow_pairs.append((pair_info, processing_time))

            i += 3  # Skip to next potential pattern
        else:
            i += 1

    # Calculate statistics for time between log lines (vocab len lines only)
    vocab_timestamps = [ts for ts in timestamps if ts is not None]
    time_intervals = []
    for i in range(1, len(vocab_timestamps)):
        interval = (vocab_timestamps[i] - vocab_timestamps[i - 1]).total_seconds()
        time_intervals.append(interval)

    # Print statistics
    print("\n" + "=" * 50)
    print("TIME BETWEEN LOG LINES STATISTICS")
    print("=" * 50)

    if time_intervals:
        print(f"Min time between log lines: {min(time_intervals):.6f} seconds")
        print(f"Max time between log lines: {max(time_intervals):.6f} seconds")
        print(
            f"Average time between log lines: {statistics.mean(time_intervals):.6f} seconds"
        )
        print(
            f"Median time between log lines: {statistics.median(time_intervals):.6f} seconds"
        )
    else:
        print("No time intervals found.")

    print("\n" + "=" * 50)
    print("VOCAB PROCESSING ANALYSIS")
    print("=" * 50)

    if processing_times:
        print(f"Total pairs processed: {len(processing_times)}")
        print(
            f"Average processing time: {statistics.mean(processing_times):.6f} seconds"
        )

    if slow_pairs:
        print(f"\nVocab pairs taking more than 1 minute to process:")
        # Optinally, sort by time
        # slow_pairs.sort(key=lambda slow_pair : slow_pair[1], reverse=True)
        for pair_info, proc_time in slow_pairs:
            minutes = proc_time / 60
            print(f"  {pair_info} -> {proc_time:.6f} seconds ({minutes:.2f} minutes)")
    else:
        print("\nNo vocab pairs took more than 1 minute to process.")

    print("=" * 50)

    return time_intervals


def create_plot(timestamps, lengths, time_intervals, output_file=None):
    """
    Create side-by-side plots of vocab length over time and time intervals between log lines.

    Args:
        timestamps (list): List of datetime objects (vocab len lines only)
        lengths (list): List of vocab lengths
        time_intervals (list): List of time intervals between consecutive log lines
        output_file (str, optional): Path to save the plot
    """
    if not timestamps or not lengths:
        print("No data to plot.")
        return

    # Calculate time from start in seconds
    start_time = timestamps[0]
    time_from_start = [(ts - start_time).total_seconds() for ts in timestamps]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Vocabulary Length Over Time
    ax1.plot(
        time_from_start, lengths, marker="o", linestyle="-", linewidth=1, markersize=2
    )
    ax1.set_title("Vocabulary Length Over Time", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time from Start (seconds)", fontsize=12)
    ax1.set_ylabel("Vocabulary Length", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time Intervals Between Log Lines
    if time_intervals:
        log_line_numbers = list(
            range(2, len(timestamps) + 1)
        )  # Start from 2 since first interval is between lines 1 and 2
        ax2.plot(
            log_line_numbers,
            time_intervals,
            marker="s",
            linestyle="-",
            linewidth=1,
            markersize=2,
            color="orange",
        )
        ax2.set_title(
            "Time Intervals Between Log Lines", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Log Line Number", fontsize=12)
        ax2.set_ylabel("Time Interval (seconds)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add stats to second plot
        min_interval = min(time_intervals)
        max_interval = max(time_intervals)
        ax2.text(
            0.02,
            0.98,
            f"Min: {min_interval:.3f}s\nMax: {max_interval:.3f}s",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Parse log file and plot vocab length over time"
    )
    parser.add_argument("filename", help="Path to the log file")
    parser.add_argument("-o", "--output", help="Output file for the plot (optional)")

    args = parser.parse_args()

    print(f"Parsing log file: {args.filename}")
    timestamps, lengths, pairs = parse_log_file(args.filename)

    # Filter out None values to get only vocab len data for plotting
    vocab_timestamps = [ts for ts in timestamps if ts is not None]
    vocab_lengths = [length for length in lengths if length is not None]

    if vocab_timestamps and vocab_lengths:
        print(f"Successfully parsed {len(vocab_timestamps)} vocab entries")

        # Analyze processing times and get time intervals
        time_intervals = analyze_processing_times(timestamps, lengths, pairs)

        # Create plots
        create_plot(vocab_timestamps, vocab_lengths, time_intervals, args.output)
    else:
        print("No valid vocab data found in the log file.")


if __name__ == "__main__":
    main()
