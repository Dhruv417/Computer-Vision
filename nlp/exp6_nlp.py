import nltk
from nltk.tokenize import (
    sent_tokenize, word_tokenize, TreebankWordTokenizer,
    RegexpTokenizer, PunktSentenceTokenizer,
    WhitespaceTokenizer, WordPunctTokenizer
)
nltk.download('punkt_tab')

file_path = "input.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

print("===== Original Text Sample =====")
print(text[:500], "\n")

print("===== Sentence Tokenization =====")
print("Default sent_tokenize (Punkt):")
print(sent_tokenize(text)[:5], "\n")  # first 5 sentences

custom_sent_tokenizer = PunktSentenceTokenizer()
print("Custom PunktSentenceTokenizer:")
print(custom_sent_tokenizer.tokenize(text)[:5], "\n")

print("===== Word Tokenization =====")
print("Default word_tokenize:")
print(word_tokenize(text)[:20], "\n")  # first 20 words

treebank_tokenizer = TreebankWordTokenizer()
print("TreebankWordTokenizer:")
print(treebank_tokenizer.tokenize(text)[:20], "\n")

regexp_tokenizer = RegexpTokenizer(r'\w+')
print("RegexpTokenizer (only words, no punctuation):")
print(regexp_tokenizer.tokenize(text)[:20], "\n")

whitespace_tokenizer = WhitespaceTokenizer()
print("WhitespaceTokenizer:")
print(whitespace_tokenizer.tokenize(text)[:20], "\n")

word_punct_tokenizer = WordPunctTokenizer()
print("WordPunctTokenizer:")
print(word_punct_tokenizer.tokenize(text)[:20], "\n")
