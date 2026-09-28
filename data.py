from datasets import load_dataset
import re
from pathlib import Path
from dotenv import load_dotenv
import os

TEST_FRACTION = 0.10
RANDOM_SEED = 42
OUT_DIR = Path("tokenizer_models_full")
OUT_DIR.mkdir(exist_ok=True)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

def load_yankari():
    print("Loading Yankari (acflp/YANKARI) from Hugging Face ...")
    ds = load_dataset("acflp/YANKARI", split="train", token=HF_TOKEN)
    # Adjust the text column name below if the dataset schema differs --
    # print(ds.column_names) to check on first run.
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    print(f"Using column '{text_col}'. Total documents: {len(ds)}")

    ds = ds.shuffle(seed=RANDOM_SEED)
    n_test = int(len(ds) * TEST_FRACTION)
    test_ds = ds.select(range(n_test))
    train_ds = ds.select(range(n_test, len(ds)))

    train_lines = [d.strip() for d in train_ds[text_col] if d and d.strip()]
    test_lines = [d.strip() for d in test_ds[text_col] if d and d.strip()]

    print(f"Train docs: {len(train_lines)} | Test docs: {len(test_lines)}")

    train_path = OUT_DIR / "yankari_train.txt"
    test_path = OUT_DIR / "yankari_test.txt"
    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    test_path.write_text("\n".join(test_lines), encoding="utf-8")

    total_train_tokens = sum(len(l.split()) for l in train_lines)
    print(f"Approx. training tokens (whitespace-split): {total_train_tokens:,}")

    return str(train_path), str(test_path), train_lines, test_lines


def whitespace_words(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)