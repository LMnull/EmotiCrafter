import argparse
import csv
from pathlib import Path


DEFAULT_PROMPT_COLUMNS = ("Neutral_Prompt", "Emotional_Prompt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mark text prompts that exceed the SDXL CLIP tokenizer limit."
    )
    parser.add_argument("--sdxl_path", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, default=Path("./data/prompt_mapping.csv"))
    parser.add_argument("--output_csv", type=Path, default=Path("./data/over_77_token_prompts.csv"))
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--prompt_columns", nargs="+", default=list(DEFAULT_PROMPT_COLUMNS))
    return parser.parse_args()


def load_sdxl_tokenizers(sdxl_path):
    from transformers import AutoTokenizer

    return [
        ("tokenizer", AutoTokenizer.from_pretrained(str(sdxl_path), subfolder="tokenizer")),
        ("tokenizer_2", AutoTokenizer.from_pretrained(str(sdxl_path), subfolder="tokenizer_2")),
    ]


def token_count(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(encoded.input_ids)


def find_overlength_prompts(csv_path, tokenizers, prompt_columns, max_length):
    marked_rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader, start=1):
            row_id = row.get("Id", row_index)
            for prompt_column in prompt_columns:
                prompt = row.get(prompt_column, "")
                if not prompt:
                    continue
                counts = {
                    tokenizer_name: token_count(tokenizer, prompt)
                    for tokenizer_name, tokenizer in tokenizers
                }
                max_count = max(counts.values())
                if max_count <= max_length:
                    continue
                marked_rows.append({
                    "row_index": row_index,
                    "Id": row_id,
                    "prompt_column": prompt_column,
                    "max_token_count": max_count,
                    "tokenizer_count": counts["tokenizer"],
                    "tokenizer_2_count": counts["tokenizer_2"],
                    "text": prompt,
                })
    return marked_rows


def write_marked_rows(output_csv, marked_rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "Id",
        "prompt_column",
        "max_token_count",
        "tokenizer_count",
        "tokenizer_2_count",
        "text",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(marked_rows)


if __name__ == "__main__":
    args = parse_args()
    if args.max_length < 1:
        raise ValueError("--max_length must be greater than 0")

    tokenizers = load_sdxl_tokenizers(args.sdxl_path)
    marked_rows = find_overlength_prompts(
        csv_path=args.csv_path,
        tokenizers=tokenizers,
        prompt_columns=args.prompt_columns,
        max_length=args.max_length,
    )
    write_marked_rows(args.output_csv, marked_rows)

    print(f"Found {len(marked_rows)} prompt(s) over {args.max_length} tokens.")
    print(f"Report saved to: {args.output_csv}")