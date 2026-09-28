import json
import time
import os
from pathlib import Path
from data import load_yankari
from tokenizerstrainers import train_bpe, train_wordpiece, train_bpe_sp, train_unigram_sp, train_morfessor
from evaluation import evaluate_hf_tokenizer, evaluate_sentencepiece, evaluate_morfessor

VOCAB_SIZES = [8000, 16000, 25000, 32000]
PRIMARY_VOCAB = 25000

RESULTS_DIR = Path("results_full")
RESULTS_DIR.mkdir(exist_ok=True)

os.environ["GLOG_minloglevel"] = "2"

def main():
    t0 = time.time()
    train_file, test_file, train_lines, test_lines = load_yankari()
    # Cap the number of test *lines* iterated for word-level UNK checks if
    # this becomes too slow on the full test split; otherwise leave as is.

    all_results = {}

    for vocab_size in VOCAB_SIZES:
        print(f"\n{'='*70}\nVOCAB SIZE = {vocab_size}\n{'='*70}")

        print("Training BPE ...")
        bpe = train_bpe(train_file, vocab_size)
        all_results.setdefault("BPE", {})[vocab_size] = evaluate_hf_tokenizer(
            bpe, test_lines, unk_tokens=("<unk>",))

        print("Training WordPiece ...")
        wp = train_wordpiece(train_file, vocab_size)
        all_results.setdefault("WordPiece", {})[vocab_size] = evaluate_hf_tokenizer(
            wp, test_lines, unk_tokens=("[UNK]",), continuation_prefix="##")

        print("Training BPE (SentencePiece) ...")
        bpesp = train_bpe_sp(train_file, vocab_size)
        all_results.setdefault("BPE/SentencePiece", {})[vocab_size] = evaluate_sentencepiece(
            bpesp, test_lines)

        print("Training Unigram (SentencePiece) ...")
        unigram = train_unigram_sp(train_file, vocab_size)
        all_results.setdefault("Unigram/SentencePiece", {})[vocab_size] = evaluate_sentencepiece(
            unigram, test_lines)

    print("\nTraining Morfessor (unsupervised, single run, no vocab param) ...")
    morf = train_morfessor(train_lines)
    morf_result = evaluate_morfessor(morf, test_lines)
    all_results["Morfessor"] = {"n/a": morf_result}

    # ------------------------------------------------------------------
    # Report: primary vocab size table
    # ------------------------------------------------------------------
    print(f"\n{'='*100}\nPRIMARY RESULTS AT VOCAB SIZE = {PRIMARY_VOCAB}\n{'='*100}")
    header = f"{'Tokenizer':<24}{'Fertility':>12}{'Chars/Token':>14}{'UNK %':>10}{'Contin. %':>12}{'WordPreserv %':>16}"
    print(header)
    print("-" * 100)
    for name in ["BPE", "WordPiece", "BPE/SentencePiece", "Unigram/SentencePiece"]:
        r = all_results[name][PRIMARY_VOCAB]
        contin = f"{r['continuation_ratio_pct']:.2f}" if r["continuation_ratio_pct"] is not None else "n/a"
        print(f"{name:<24}{r['fertility']:>12.3f}{r['compression_rate_chars_per_token']:>14.3f}"
              f"{r['unk_rate_pct']:>10.3f}{contin:>12}{r['word_boundary_preservation_pct']:>16.2f}")
    r = all_results["Morfessor"]["n/a"]
    print(f"{'Morfessor':<24}{r['fertility']:>12.3f}{r['compression_rate_chars_per_token']:>14.3f}"
          f"{r['unk_rate_pct']:>10.3f}{'n/a':>12}{r['word_boundary_preservation_pct']:>16.2f}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Report: vocab-size sweep (fertility trend)
    # ------------------------------------------------------------------
    print("\nFertility across vocab-size sweep:")
    print(f"{'Vocab size':<12}{'BPE':>10}{'WordPiece':>12}{'BPESP':>12}{'Unigram':>10}")
    for v in VOCAB_SIZES:
        print(f"{v:<12}{all_results['BPE'][v]['fertility']:>10.3f}"
              f"{all_results['WordPiece'][v]['fertility']:>12.3f}"
              f"{all_results['BPE/SentencePiece'][v]['fertility']:>12.3f}"
              f"{all_results['Unigram/SentencePiece'][v]['fertility']:>10.3f}")
    print(f"Morfessor (reference, no vocab param): {morf_result['fertility']:.3f}")

    with open(RESULTS_DIR / "full_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} minutes")
    print(f"Saved detailed results to {RESULTS_DIR / 'full_sweep_results.json'}")


if __name__ == "__main__":
    main()