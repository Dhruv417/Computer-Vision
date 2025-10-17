mport sys
import re
from pathlib import Path
from typing import Tuple, List

# NLTK for tokenization and edit distance (Levenshtein)
try:
    from nltk import word_tokenize, download as nltk_download
    from nltk.metrics.distance import edit_distance
    _NLTK_OK = True
except Exception:  # Fallback if NLTK is not installed
    _NLTK_OK = False

# scikit-learn for vectorizers and cosine similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure UTF-8 capable stdout on Windows terminals
try:
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
except Exception:
    pass


def _ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are available without interactive prompts."""
    if not _NLTK_OK:
        return
    try:
        # punkt is needed for word_tokenize
        nltk_download('punkt', quiet=True)
        # punkt_tab is required for newer NLTK versions
        try:
            nltk_download('punkt_tab', quiet=True)
        except Exception:
            pass
    except Exception:
        # Do not fail the whole script if downloads are blocked
        pass


def read_two_texts_from_file(file_path: Path) -> Tuple[str, str]:
    """Read two texts from a file with flexible parsing.

    Parsing strategies, in order:
    - Marked lines: lines beginning with "text1:"/"text 1:" and "text2:"/"text 2:" (case-insensitive)
    - Paragraphs separated by a blank line (first two paragraphs)
    - First two non-empty lines
    - If a single line contains a delimiter "|||", split around it
    """
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    # Strip potential BOM if present
    if content.startswith('\ufeff'):
        content = content.lstrip('\ufeff')
    if not content.strip():
        raise ValueError("Input file is empty. Provide two texts in the file.")

    lines = [line.rstrip('\n') for line in content.splitlines()]

    # Strategy 1: Marked blocks (support multi-line paragraphs per section)
    # Example:
    # Text1: First line for text 1
    # More lines...
    #
    # Text2: First line for text 2
    # More lines...
    text1_block: List[str] = []
    text2_block: List[str] = []
    current_label: str = ''
    for raw in lines:
        stripped = raw.strip()
        low = stripped.lower()
        if low.startswith(('text1:', 'text 1:')):
            current_label = 'text1'
            after = raw.split(':', 1)[1]
            if after.strip():
                text1_block.append(after.strip())
            continue
        if low.startswith(('text2:', 'text 2:')):
            current_label = 'text2'
            after = raw.split(':', 1)[1]
            if after.strip():
                text2_block.append(after.strip())
            continue
        if current_label == 'text1':
            text1_block.append(raw)
        elif current_label == 'text2':
            text2_block.append(raw)
    if text1_block and text2_block:
        return '\n'.join(text1_block).strip(), '\n'.join(text2_block).strip()

    # Strategy 2: Paragraphs separated by blank lines
    paragraphs: List[str] = []
    current: List[str] = []
    for line in lines:
        if line.strip() == '':
            if current:
                paragraphs.append('\n'.join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append('\n'.join(current).strip())
    if len(paragraphs) >= 2:
        return paragraphs[0], paragraphs[1]

    # Strategy 3: First two non-empty lines
    non_empty = [ln.strip() for ln in lines if ln.strip()]
    if len(non_empty) >= 2:
        return non_empty[0], non_empty[1]

    # Strategy 4: Single-line with explicit delimiter
    if '|||' in content:
        left, right = content.split('|||', 1)
        return left.strip(), right.strip()

    raise ValueError(
        "Could not find two texts in the file. Use two lines, two paragraphs, or mark with 'Text1:' / 'Text2:' or use '|||'."
    )


def normalize_for_jaccard(text: str) -> List[str]:
    """Tokenize and normalize text for Jaccard similarity using NLTK when available."""
    if _NLTK_OK:
        try:
            tokens = word_tokenize(text)
        except LookupError:
            _ensure_nltk_resources()
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = re.findall(r"\b\w+\b", text)
    else:
        tokens = re.findall(r"\b\w+\b", text)

    # Lowercase and keep alphanumerics
    normalized = [t.lower() for t in tokens if re.search(r"\w", t)]
    return normalized


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(normalize_for_jaccard(text_a))
    tokens_b = set(normalize_for_jaccard(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def levenshtein_distance(text_a: str, text_b: str) -> int:
    if _NLTK_OK:
        try:
            return edit_distance(text_a, text_b)
        except Exception:
            pass
    # Minimal fallback implementation if NLTK unavailable
    return _levenshtein_fallback(text_a, text_b)


def _levenshtein_fallback(a: str, b: str) -> int:
    # Iterative DP with O(min(n, m)) memory
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def cosine_similarities(text_a: str, text_b: str) -> Tuple[float, float]:
    corpus = [text_a, text_b]

    # CountVectorizer cosine similarity
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(corpus)
    count_cosine = float(cosine_similarity(count_matrix[0], count_matrix[1])[0][0])

    # TfidfVectorizer cosine similarity
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(corpus)
    tfidf_cosine = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

    return count_cosine, tfidf_cosine


def main() -> None:
    file_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('file.txt')
    if not file_arg.exists():
        raise FileNotFoundError(f"Input file not found: {file_arg}")

    _ensure_nltk_resources()

    text1, text2 = read_two_texts_from_file(file_arg)

    lev = levenshtein_distance(text1, text2)
    jac = jaccard_similarity(text1, text2)
    cos_count, cos_tfidf = cosine_similarities(text1, text2)

    print("Input file:", str(file_arg))
    print("Text 1:", text1)
    print("Text 2:", text2)
    print("-")
    print(f"Levenshtein distance (NLTK edit_distance): {lev}")
    print(f"Jaccard similarity (token set): {jac:.4f}")
    print(f"Cosine similarity (CountVectorizer): {cos_count:.4f}")
    print(f"Cosine similarity (TfidfVectorizer): {cos_tfidf:.4f}")


if __name__ == '__main__':
    main()