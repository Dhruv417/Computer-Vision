import nltk
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('nps_chat')
nltk.download('movie_reviews')
nltk.download('reuters')
nltk.download('udhr')

from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown, gutenberg, nps_chat, movie_reviews, reuters, udhr

def analyze_corpus(words, title):
    print(f"\n🔍 Analyzing: {title}")

    # Convert to lowercase for normalization
    words = [w.lower() for w in words if w.isalpha()]

    fdist = FreqDist(words)
    most_common = fdist.most_common(10)
    print("Top 10 Words:", most_common)

    fdist.plot(20, title=f"{title} - Frequency Distribution")

    # Zipf's Law: Plot rank vs frequency (log-log)
    freqs = np.array([freq for word, freq in fdist.most_common()])
    ranks = np.arange(1, len(freqs) + 1)

    plt.figure(figsize=(6,4))
    plt.loglog(ranks, freqs)
    plt.title(f"Zipf's Law: {title}")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.grid(True)
    plt.show()

analyze_corpus(brown.words(), "Brown Corpus")

analyze_corpus(gutenberg.words('austen-emma.txt'), "Gutenberg - Emma")

chat_words = [word for post in nps_chat.posts() for word in post]
analyze_corpus(chat_words, "NPS Chat Corpus")

analyze_corpus(movie_reviews.words(), "Movie Reviews")

analyze_corpus(reuters.words(), "Reuters Corpus")

analyze_corpus(udhr.words('English-Latin1'), "UDHR - English")

import nltk
from nltk.corpus import gutenberg
import matplotlib.pyplot as plt

nltk.download('gutenberg')

# Get lengths of texts
lengths = {fileid: len(gutenberg.words(fileid)) for fileid in gutenberg.fileids()}

# Sort by length (descending)
sorted_lengths = dict(sorted(lengths.items(), key=lambda item: item[1], reverse=True))

# Print sorted lengths
print("📏 Gutenberg Texts by Word Count (Descending):\n")
for i, (fileid, length) in enumerate(sorted_lengths.items(), 1):
    print(f"{i:>2}. {fileid:<30} : {length} words")

# Bar plot
plt.figure(figsize=(12,6))
plt.bar(sorted_lengths.keys(), sorted_lengths.values(), color='salmon')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Word Count")
plt.title("Gutenberg Corpus - Text Length Comparison")
plt.tight_layout()
plt.show()

