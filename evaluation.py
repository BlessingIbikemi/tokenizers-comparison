from data import whitespace_words

def evaluate_hf_tokenizer(tok, test_lines, unk_tokens, continuation_prefix="##"):
    total_tokens = total_words = total_chars = unk_count = 0
    continuation_count = words_preserved = 0
    for line in test_lines:
        words = whitespace_words(line)
        total_words += len(words)
        total_chars += len(line)
        pieces = tok.encode(line).tokens
        total_tokens += len(pieces)
        unk_count += sum(1 for p in pieces if p in unk_tokens)
        continuation_count += sum(1 for p in pieces if p.startswith(continuation_prefix))
        for w in words:
            if len(tok.encode(w).tokens) == 1:
                words_preserved += 1
    return _pack(total_tokens, total_words, total_chars, unk_count,
                 continuation_count, words_preserved)

def evaluate_sentencepiece(sp, test_lines):
    total_tokens = total_words = total_chars = unk_count = 0
    continuation_count = words_preserved = 0
    unk_id = sp.unk_id()
    for line in test_lines:
        words = whitespace_words(line)
        total_words += len(words)
        total_chars += len(line)
        ids = sp.encode(line, out_type=int)
        pieces = sp.encode(line, out_type=str)
        total_tokens += len(pieces)
        unk_count += sum(1 for i in ids if i == unk_id)
        continuation_count += sum(1 for p in pieces if not p.startswith("\u2581"))
        for w in words:
            if len(sp.encode(w, out_type=str)) == 1:
                words_preserved += 1
    return _pack(total_tokens, total_words, total_chars, unk_count,
                 continuation_count, words_preserved)


def evaluate_morfessor(model, test_lines):
    total_tokens = total_words = total_chars = words_preserved = 0
    for line in test_lines:
        words = whitespace_words(line)
        total_words += len(words)
        total_chars += len(line)
        for w in words:
            segments = model.viterbi_segment(w)[0] if any(c.isalpha() for c in w) else [w]
            total_tokens += len(segments)
            if len(segments) == 1:
                words_preserved += 1
    return _pack(total_tokens, total_words, total_chars, 0, None, words_preserved)


def _pack(total_tokens, total_words, total_chars, unk_count, continuation_count, words_preserved):
    return {
        "fertility": total_tokens / total_words,
        "compression_rate_chars_per_token": total_chars / total_tokens,
        "unk_rate_pct": 100 * unk_count / total_tokens,
        "continuation_ratio_pct": (100 * continuation_count / total_tokens
                                    if continuation_count is not None else None),
        "word_boundary_preservation_pct": 100 * words_preserved / total_words,
        "total_tokens": total_tokens,
        "total_words": total_words,
    }