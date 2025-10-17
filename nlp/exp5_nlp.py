
import nltk
from collections import defaultdict
from nltk.corpus import brown

nltk.download('brown')
def build_bigram_table_from_corpus(corpus_words):
    vocab = set(corpus_words)
    V = len(vocab)

    unigram_counts = defaultdict(int)
    for word in corpus_words:
        unigram_counts[word] += 1

    bigram_counts = defaultdict(int)
    for i in range(len(corpus_words) - 1):
        bigram = (corpus_words[i], corpus_words[i+1])
        bigram_counts[bigram] += 1

    bigram_probs = {}
    for w1 in vocab:
        for w2 in vocab:
            count_w1_w2 = bigram_counts[(w1, w2)]
            prob = (count_w1_w2 + 1) / (unigram_counts[w1] + V)
            bigram_probs[(w1, w2)] = prob

    return bigram_probs, vocab

words = brown.words()[:1000]
bigram_probs, vocab = build_bigram_table_from_corpus([w.lower() for w in words])

print("Vocabulary Size:", len(vocab))
print("\nSample smoothed bigram probabilities:\n")

# Print first 15 bigram probabilities
for bigram, prob in list(bigram_probs.items())[:15]:
    print(f"P({bigram[1]} | {bigram[0]}) = {prob:.5f}")