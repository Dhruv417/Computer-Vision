#Exp 4
import nltk
from nltk.corpus import brown
from nltk import ngrams, FreqDist

nltk.download('brown')

words = [w.lower() for w in brown.words() if w.isalpha()]

bigrams = list(ngrams(words, 2))
bigram_freq = FreqDist(bigrams)

print("\n Top 20 Bigrams in Brown Corpus:")
for bigram, freq in bigram_freq.most_common(20):
    print(f"{bigram} : {freq}")

trigrams = list(ngrams(words, 3))
trigram_freq = FreqDist(trigrams)

print("\n Top 20 Trigrams in Brown Corpus:")
for trigram, freq in trigram_freq.most_common(20):
    print(f"{trigram} : {freq}")
