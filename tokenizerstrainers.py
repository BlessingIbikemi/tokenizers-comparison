from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import sentencepiece as spm
import morfessor
from pathlib import Path
import os
from data import whitespace_words

OUT_DIR = Path("tokenizer_models_full")
OUT_DIR.mkdir(exist_ok=True)

def train_bpe(train_file, vocab_size):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
        show_progress=True,
    )
    tok.train([train_file], trainer)
    tok.save(str(OUT_DIR / f"bpe_{vocab_size}.json"))
    return tok


def train_wordpiece(train_file, vocab_size):
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
        continuing_subword_prefix="##",
        show_progress=True,
    )
    tok.train([train_file], trainer)
    tok.save(str(OUT_DIR / f"wordpiece_{vocab_size}.json"))
    return tok

def train_bpe_sp(train_file, vocab_size):
    model_prefix = f"bpe-sp{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=str(OUT_DIR / model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        num_threads=os.cpu_count() or 4,
        unk_id=0, pad_id=1, bos_id=2, eos_id=3,
    )
    sp = spm.SentencePieceProcessor()
    sp.load(str(OUT_DIR / f"{model_prefix}.model"))
    return sp

def train_unigram_sp(train_file, vocab_size):
    model_prefix = f"unigram_{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=str(OUT_DIR / model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,   # important for 30M+ token corpora
        num_threads=os.cpu_count() or 4,
        unk_id=0, pad_id=1, bos_id=2, eos_id=3,
    )
    sp = spm.SentencePieceProcessor()
    sp.load(str(OUT_DIR / f"{model_prefix}.model"))
    return sp


def train_morfessor(train_lines):
    word_counts = {}
    for line in train_lines:
        for w in whitespace_words(line):
            if any(ch.isalpha() for ch in w):
                word_counts[w] = word_counts.get(w, 0) + 1
    train_data = [(c, w) for w, c in word_counts.items()]
    model = morfessor.BaselineModel()
    model.load_data(train_data)
    model.train_batch()
    io = morfessor.MorfessorIO()
    io.write_binary_model_file(str(OUT_DIR / "morfessor.bin"), model)
    print(f"Morfessor trained on {len(train_data):,} unique word types "
          f"(effective vocabulary: {len(model.get_constructions())} morphs)")
    return model
