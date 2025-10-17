#Exp 2
import nltk
from nltk.corpus import brown, stopwords
from collections import Counter
import string

nltk.download('brown')
nltk.download('stopwords')

words = [word.lower() for word in brown.words()]

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]

word_counts = Counter(filtered_words)

most_common_words = word_counts.most_common(20)

print("Most Frequent Words in Brown Corpus (excluding stopwords & punctuation):")
for word, freq in most_common_words:
    print(f"{word}: {freq}")
