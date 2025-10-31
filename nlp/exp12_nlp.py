import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser

with open(r"C:\Users\Dhruv\OneDrive\Desktop\New folder\nlp\input.txt", "r", encoding="utf-8") as file:
    sentence = file.read().strip()

# Tokenize and tag
words = word_tokenize(sentence)
tagged = pos_tag(words)

print("POS Tags:")
print(tagged)

# Define proper chunk + chink grammar
chunk_grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}    # chunk noun phrases
        }<VB.*|IN>+{           # chink (remove) verbs and prepositions
"""

chunk_parser = RegexpParser(chunk_grammar)
chunked_output = chunk_parser.parse(tagged)

print("\nChunked and Chinked Output:")
print(chunked_output)

chunked_output.draw()
