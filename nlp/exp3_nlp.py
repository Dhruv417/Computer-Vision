#Exp 3
import nltk
from nltk.corpus import brown

nltk.download('brown', quiet=True)

words = list(brown.words())

occurrences = []

for i, word in enumerate(words):
    if word.lower() == "the":
        prev_word = words[i-1] if i > 0 else None
        next_word = words[i+1] if i < len(words)-1 else None
        occurrences.append((i, prev_word, word, next_word))

print("Positions and context of 'the':")
for pos, prev, current, nextw in occurrences[:20]:
    print(f"Index {pos}: {prev} {current} {nextw}")

print(f"\nTotal occurrences of 'the': {len(occurrences)}")