import argparse
import json
from pathlib import Path


def load_merge_lookup(path: Path, key_field: str) -> dict:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)

    return {
        record[key_field]: record
        for record in records
        if isinstance(record, dict) and key_field in record
    }


def split_json(
    input_path: Path,
    output_dir: Path,
    key_field: str = "kayit_id",
    merge_lookup: dict | None = None,
) -> int:
    with input_path.open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list of objects")

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        name = record.get(key_field) if isinstance(record, dict) else None
        if merge_lookup and name in merge_lookup:
            record = {**record, **merge_lookup[name]}

        file_name = f"{name}.json" if name else f"{index:06d}.json"

        with (output_dir / file_name).open("w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split one or more JSON files containing a list of objects into separate JSON files."
    )
    parser.add_argument("inputs", nargs="+", help="One or more JSON files to split")
    parser.add_argument("output", help="Output folder for the split JSON files")
    parser.add_argument("--key", default="kayit_id", help="Field used to name each output file")
    parser.add_argument(
        "--merge",
        help="Optional JSON file whose records are merged in by --key (e.g. ground-truth labels)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    merge_lookup = load_merge_lookup(Path(args.merge), args.key) if args.merge else None

    total = 0
    for input_arg in args.inputs:
        count = split_json(Path(input_arg), output_dir, args.key, merge_lookup)
        total += count
        print(f"{input_arg}: {count} kayit '{output_dir}' klasorune yazildi.")

    print(f"Toplam {total} kayit islendi.")
