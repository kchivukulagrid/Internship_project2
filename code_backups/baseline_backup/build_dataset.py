import os
import json

from src.data_loader import load_conll2003_local
from src.preprocess import convert_example


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    print("Loading raw dataset...")
    dataset = load_conll2003_local()

    label_names = dataset["train"].features["ner_tags"].feature.names

    print("Converting train split...")
    train_data = [
        convert_example(example, label_names)
        for example in dataset["train"]
    ]

    print("Converting validation split...")
    val_data = [
        convert_example(example, label_names)
        for example in dataset["validation"]
    ]

    print("Converting test split...")
    test_data = [
        convert_example(example, label_names)
        for example in dataset["test"]
    ]

    # Create processed folder automatically
    os.makedirs("data/processed", exist_ok=True)

    save_jsonl(train_data, "data/processed/train.jsonl")
    save_jsonl(val_data, "data/processed/val.jsonl")
    save_jsonl(test_data, "data/processed/test.jsonl")

    print("Dataset successfully built and saved in data/processed/")


if __name__ == "__main__":
    main()
