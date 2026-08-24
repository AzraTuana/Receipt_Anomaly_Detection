import argparse
import json
from pathlib import Path


def split_json(input_path: Path, output_dir: Path, key_field: str = "kayit_id") -> int:
    with input_path.open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list of objects")

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        name = record.get(key_field) if isinstance(record, dict) else None
        file_name = f"{name}.json" if name else f"{index:06d}.json"

        with (output_dir / file_name).open("w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a JSON file containing a list of objects into separate JSON files."
    )
    parser.add_argument("input", nargs="?", default=r"C:\Users\azrat\AppData\Local\Temp\test_etiket.json")
    parser.add_argument("output", nargs="?", default="json_files")
    parser.add_argument("--key", default="kayit_id", help="Field used to name each output file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    count = split_json(input_path, output_dir, args.key)
    print(f"{count} kayit '{output_dir}' klasorune ayri json dosyalari olarak yazildi.")
