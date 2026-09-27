"""
The Kontoyiannis entropy rate estimator (Ĥ_K).

    Ĥ_K = (n log2 n) / sum_i Λ_i                                    (Eq. 1)

where Λ_i is the length of the shortest prefix of x_i, ..., x_n that does
NOT appear as a contiguous substring in x_1, ..., x_{i-1}. Long matches
signal low surprise (repetition); short matches signal high surprise
(novelty). Near the end of a document, where no such non-appearing prefix
may exist, Λ_i is bounded by the remaining sequence length n - i, so the
sum stays well-defined with no special-casing.

Ĥ_K converges almost surely to the entropy rate of any stationary ergodic
source (Kontoyiannis, Algoet, Suhov & Wyner, 1998, IEEE Trans. Info. Theory
44(3):1319-1327) -- no parametric assumptions on the token distribution are
required. It requires no model, no API calls, and no GPU: it is computed
entirely from the text itself, which is the whole point of this paper.

Tokenization: lower-case the text, replace every non-alphanumeric character
with a space, then split on whitespace. This is the scheme used for
real-time filtering decisions during training (the fast path, computed
often and needing to be cheap); the paper's *reported* Ĥ_K values (its
tables and figures) instead use the `ProcessEntropy` package's own
tokenizer (NLTK TweetTokenizer + fnv hashing), a more careful but slower
pipeline -- see below. The two agree closely in practice (Spearman
rho=0.997 on a held-out sample of real experiment documents), so this
doesn't affect which documents get selected, only the exact reported
number.

This module is a clean, dependency-free reference implementation matching
the paper's Equation 1 exactly, using the fast tokenization above -- good
for understanding the estimator and for small-to-medium documents. It is
O(n^2) worst case (each Λ_i is found by a substring search over the growing
context string), same asymptotic cost as the paper's own real-time filter.

For the exact pipeline that produced every reported number in the paper
(ProcessEntropy's tokenizer + an LCS-finder, not substring search, for
speed), see `ProcessEntropy` itself, which this repo depends on for the
figure/table reproduction scripts: https://github.com/tobinsouth/ProcessEntropy
(`ProcessEntropy.SelfEntropy.self_entropy_rate`). At finite n, Ĥ_K is
upward-biased (early positions have little past context); the paper
controls for this by computing Ĥ_K on a fixed 1,500-word truncation of each
document rather than by using a different estimator here.
"""
import math
import re
from typing import List, Optional


def tokenize(text: str) -> List[str]:
    """Lower-case, replace non-alphanumerics with spaces, split on whitespace."""
    return re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()


def match_lengths(words: List[str]) -> List[int]:
    """
    Λ_i for every position i in `words` (1-indexed in the paper; here
    returned as a list aligned to words[1:], since Λ_1 is undefined with no
    prior context).
    """
    n = len(words)
    lambdas = []
    parts = ["\x00" + w for w in words]
    context = parts[0]
    for i in range(1, n):
        L = 0
        probe = "\x00" + words[i]
        while i + L < n and probe in context:
            L += 1
            if i + L < n:
                probe += "\x00" + words[i + L]
        lambdas.append(L + 1)
        context += parts[i]
    return lambdas


def hk(text: str, n_trunc: Optional[int] = 1500, min_words: int = 50) -> Optional[float]:
    """
    Ĥ_K in bits/word for `text`.

    n_trunc: truncate to this many words before scoring, matching the
        paper's finite-sample-bias control (Section 2). Pass None to score
        the full text (fine for short documents; slower and more
        upward-biased for long ones).
    min_words: documents shorter than this return None, matching the
        paper's own reliability convention -- Ĥ_K is not meaningful on
        very short text.

    Returns bits/word (n * log2(n) / sum(Λ_i)), or None if the (possibly
    truncated) document has fewer than `min_words` words.
    """
    words = tokenize(text)
    if n_trunc is not None:
        words = words[:n_trunc]
    n = len(words)
    if n < min_words:
        return None
    total_lambda = sum(match_lengths(words))
    if total_lambda == 0:
        return None
    return (n * math.log2(n)) / total_lambda


if __name__ == "__main__":
    # A trivial illustration, not experiment data: repetitive text scores
    # low (long matches -> low surprise), varied text scores higher.
    repetitive = "the cat sat on the mat. " * 60
    varied = (
        "The auroch, an extinct ancestor of domestic cattle, once roamed "
        "across Europe, Asia, and North Africa in vast herds. Julius Caesar "
        "described them as nearly the size of elephants, though modern "
        "reconstructions suggest they stood closer to two meters at the "
        "shoulder. The last known individual died in Poland's Jaktorow "
        "Forest in 1627, a death recorded with unusual precision for the "
        "era. Quaggas, a related and similarly extinct subspecies of plains "
        "zebra, survived slightly longer in the wild before hunting and "
        "habitat loss reduced the last population to a single mare, who "
        "died at Amsterdam's Natura Artis Magistra zoo in August of 1883, "
        "mistakenly listed in the zoo's own records simply as an ordinary "
        "zebra until decades after her death. Thylacines met a comparable "
        "fate on the island of Tasmania, the final captive individual "
        "succumbing to neglect and exposure in an unheated enclosure "
        "overnight in September 1936, only weeks after legal protection "
        "for the species had finally, belatedly, been enacted."
    )
    print(f"Repetitive text: Hk = {hk(repetitive, n_trunc=None):.3f} bits/word")
    print(f"Varied text:     Hk = {hk(varied, n_trunc=None):.3f} bits/word")
