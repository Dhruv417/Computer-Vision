import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import words
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get standard English vocabulary
english_vocab = set(w.lower() for w in words.words())

def find_unusual_words(text):
    """Return words not found in the English vocabulary (after lemmatization)."""
    text_vocab = set(
        lemmatizer.lemmatize(w.lower()) 
        for w in word_tokenize(text) if w.isalpha()
    )
    unusual = text_vocab - english_vocab
    return sorted(unusual)

# ----------------------------------------
# Scenario 1: Plain English Text (5 lines)
# ----------------------------------------
text1 = """
Natural Language Processing is a fascinating field of computer science.
It enables machines to understand and generate human language effectively.
Researchers use algorithms to analyze syntax, semantics, and context.
NLP applications include chatbots, translators, and sentiment analysis.
This branch of AI continues to evolve with new techniques each year.
"""

# ----------------------------------------
# Scenario 2: Foreign Language Text (5 lines)
# ----------------------------------------
text2 = """
Bonjour! Je m'appelle Tanisha et j'étudie l'intelligence artificielle.
La technologie change le monde à une vitesse incroyable.
Les chercheurs développent des systèmes intelligents et adaptatifs.
J'aime apprendre de nouvelles choses et comprendre les langues humaines.
L'avenir de l'IA semble très prometteur pour notre société.
"""

# ----------------------------------------
# Scenario 3: Social Media Slang Text (5 lines)
# ----------------------------------------
text3 = """
OMG this AI project is literally blowing my mind rn!
The output is so dope, I can’t even explain it lol.
People be like “bruh that bot smarter than me fr”.
Tbh this tech is lowkey scary but also super cool ngl.
Can’t wait to flex these results on my Insta story 😂.
"""

# Process and print unusual words for each scenario
for i, (title, text) in enumerate([
    ("Scenario 1: Plain English Text", text1),
    ("Scenario 2: Foreign Language Text", text2),
    ("Scenario 3: Social Media Slang Text", text3)
], start=1):
    unusual = find_unusual_words(text)
    print(f"{title}")
    print("Unusual Words:", unusual)
    print("-" * 70)       