"""
Recomputes the per-document summary behind Table 1 and Figure 1 (the main
filtering-experiment result) from the full raw document corpus.

This script is NOT needed to reproduce the paper's figures/tables -- the
output it produces is already bundled at results/phase5_summary.json (text
stripped out, numeric fields only), which make_figure.py reads directly.
It's included so the summary itself is reproducible from the ground truth:
if you want to verify it, or recompute Hk under a different truncation
length, download the full raw corpus (1.6 GB, all conditions/generations,
full document text) from:

    https://huggingface.co/datasets/lewismath/phase5-raw

and point --raw-dir at it.

Uses the exact same computation as the paper's own paper/make_figures.py:
ProcessEntropy's tokenizer (NLTK TweetTokenizer + fnv hashing) and
self_entropy_rate, truncated to the first 1,500 tokens per document. This
is the "reported" pipeline, not the faster whitespace-tokenized version
used for real-time filtering decisions during training (see hk/estimator.py
for that one, and its docstring for why the two agree closely but aren't
identical).
"""
import argparse, json, os

N_TRUNC = 1500
CONDITIONS = ["unfiltered", "hl_filter", "hk_filter"]


def text_metrics(text):
    """Distinct-3, Vocabulary, Rep-4 (and TTR@500, Distinct-2 as a bonus) --
    the text-diversity metrics behind Figure 1(b) and Table 1's diversity
    columns. Matches paper/make_figures.py's text_metrics() exactly (plain
    lower-case + whitespace split, no punctuation stripping -- yet another
    distinct tokenization from the two used for Hk itself; see
    hk/estimator.py's docstring for why there are three in this codebase)."""
    words = text.lower().split()
    n = len(words)
    if n < 10:
        return {}
    w500 = words[:500]
    ttr = len(set(w500)) / len(w500) if len(w500) >= 500 else None
    bigrams   = [tuple(words[i:i+2]) for i in range(n - 1)]
    trigrams  = [tuple(words[i:i+3]) for i in range(n - 2)]
    fourgrams = [tuple(words[i:i+4]) for i in range(n - 3)]
    return {
        "ttr":   ttr,
        "dist2": len(set(bigrams)) / len(bigrams) if bigrams else 0,
        "dist3": len(set(trigrams)) / len(trigrams) if trigrams else 0,
        "rep4":  1.0 - len(set(fourgrams)) / len(fourgrams) if fourgrams else 0,
        "vocab": len(set(words)),
    }


def compute_hk(text, processentropy):
    tokens = processentropy["Preprocessing"].custom_tokenize(text)[:N_TRUNC]
    if len(tokens) < 10:
        return None
    return processentropy["SelfEntropy"].self_entropy_rate(
        [processentropy["Preprocessing"].fnv(t) for t in tokens]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True,
                    help="Directory containing phase5_gen0_docs.json and "
                         "phase5_{condition}_gen{N}_docs.json (the full raw corpus).")
    p.add_argument("--output", default="results/phase5_summary.json")
    args = p.parse_args()

    from ProcessEntropy import SelfEntropy, Preprocessing
    pe = {"SelfEntropy": SelfEntropy, "Preprocessing": Preprocessing}

    with open(os.path.join(args.raw_dir, "phase5_gen0_docs.json")) as f:
        gen0_docs = json.load(f)

    records = []
    for cond in CONDITIONS:
        docs = list(gen0_docs)
        for g in range(1, 7):
            path = os.path.join(args.raw_dir, f"phase5_{cond}_gen{g}_docs.json")
            with open(path) as f:
                docs.extend(json.load(f))
        for d in docs:
            records.append({
                "condition":  cond,
                "generation": d["generation"],
                "domain":     d["domain"],
                "topic":      d["topic"],
                "doc_id":     d["doc_id"],
                "hk":         compute_hk(d["text"], pe),
                "mean_h_l":   d.get("mean_h_l"),
                "n_tokens":   d.get("n_tokens"),
                **text_metrics(d["text"]),
            })
        print(f"{cond}: {len(docs)} documents processed")

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
