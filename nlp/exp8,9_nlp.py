#exp 8,9
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.wsd import lesk
from nltk.stem import PorterStemmer
import spacy

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------ Read text from file ------------------
with open("input.txt", "r") as f:   # change filename as needed
    text = f.read()

print("Original Text:\n", text)

tokens = nltk.word_tokenize(text)

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
filtered_tokens = [ps.stem(w.lower()) for w in tokens if w.lower() not in stop_words]

print("\nTokens after Stopword Removal & Stemming:\n", filtered_tokens)

pos_tags = nltk.pos_tag(tokens)
print("\nPOS Tags:\n", pos_tags)

def filtered_lesk(word, sentence):
    synset = lesk(sentence, word)
    if not synset:
        return None

    wrong_senses = {
        "Amazon": "parrot",
        "POS": "post office",
        "deep": "affecting one deeply",
    }

    if word in wrong_senses and wrong_senses[word] in synset.definition().lower():
        candidates = wn.synsets(word)
        for cand in candidates:
            if any(kw in cand.definition().lower() for kw in
                   ["language", "computer", "linguistic", "ai", "technology", "company"]):
                return cand
        return None
    return synset

print("\nWord Sense Disambiguation (Filtered Lesk Algorithm):")
sentence_tokens = nltk.word_tokenize(text)
for word in set(sentence_tokens):
    sense = filtered_lesk(word, sentence_tokens)
    if sense:
        print(f"{word} -> {sense.definition()}")

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

entities = [(ent.text, ent.label_) for ent in doc.ents]

# Fix NER for languages
def fix_ner(entities):
    corrected = []
    language_list = ["English", "Spanish", "German", "Hindi", "French", "Chinese", "Japanese"]
    for text, label in entities:
        if text in language_list:
            corrected.append((text, "LANGUAGE"))
        else:
            corrected.append((text, label))
    return corrected

corrected_entities = fix_ner(entities)

print("\nNamed Entities (Corrected NER):")
for ent, label in corrected_entities:
    print(f"{ent} --> {label}")