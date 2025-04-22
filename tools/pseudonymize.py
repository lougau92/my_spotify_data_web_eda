#!/usr/bin/env python3
import argparse
from faker import Faker
import json
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path

# Global mappings to ensure consistency across all files
artist_mapping = {}
track_mapping = {}


def setup_faker(seed=4321):
    Faker.seed(seed)
    return Faker()


def pseudonymize_entry(entry, fake):
    # Pseudonymize artist (consistent across all instances)
    original_artist = entry["artistName"]
    if original_artist not in artist_mapping:
        artist_mapping[original_artist] = fake.name()
    entry["artistName"] = artist_mapping[original_artist]

    # Pseudonymize track (consistent across all instances)
    original_track = entry["trackName"]
    if original_track not in track_mapping:
        track_mapping[original_track] = fake.catch_phrase().title()
    entry["trackName"] = track_mapping[original_track]

    # Pseudonymize msPlayed (±20% variation)
    entry["msPlayed"] = int(entry["msPlayed"] * random.uniform(0.8, 1.2))

    # Pseudonymize timestamp (while keeping order)
    original_time = datetime.strptime(entry["endTime"], "%Y-%m-%d %H:%M")
    time_offset = timedelta(days=random.randint(-10, 10), hours=random.randint(-3, 3))
    entry["endTime"] = (original_time + time_offset).strftime("%Y-%m-%d %H:%M")

    return entry


def process_files(file_paths, fake, encoding="utf-8"):
    pseudonymized_data = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
                pseudonymized_data.extend(
                    [pseudonymize_entry(entry, fake) for entry in data]
                )
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                pseudonymized_data.extend(
                    [pseudonymize_entry(entry, fake) for entry in data]
                )
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)
            sys.exit(1)
    return pseudonymized_data


def main():
    parser = argparse.ArgumentParser(
        description="Pseudonymize Spotify streaming history files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Path(s) to JSON files containing streaming history",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default=".",
        help="Output folder path for pseudonymized data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4321,
        help="Random seed for reproducible pseudonymization",
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="File encoding to use (default: utf-8)"
    )
    parser.add_argument(
        "--save-mappings",
        action="store_true",
        help="Save artist and track mappings to separate JSON files",
    )

    args = parser.parse_args()

    # Verify input files exist
    for file_path in args.input_files:
        if not Path(file_path).exists():
            print(f"Error: Input file not found: {file_path}", file=sys.stderr)
            sys.exit(1)

    # Create output folder if it doesn't exist
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fake = setup_faker(args.seed)
    random.seed(args.seed)

    print(f"Processing {len(args.input_files)} file(s)...")

    # Process each file individually and save with _pseudonymized suffix
    for input_file in args.input_files:
        input_path = Path(input_file)
        output_file = (
            output_folder / f"{input_path.stem}_pseudonymized{input_path.suffix}"
        )

        pseudonymized_data = process_files([input_file], fake, args.encoding)

        print(f"Saving pseudonymized data to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pseudonymized_data, f, indent=2, ensure_ascii=False)

    if args.save_mappings:
        print("Saving mappings...")
        with open(output_folder / "artist_mapping.json", "w", encoding="utf-8") as f:
            json.dump(artist_mapping, f, indent=2, ensure_ascii=False)
        with open(output_folder / "track_mapping.json", "w", encoding="utf-8") as f:
            json.dump(track_mapping, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
