import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')


corpus = """
Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and generate human language. It combines computational linguistics with machine learning and deep learning models. As a result, machines can perform tasks such as translation, summarization, sentiment analysis, and question answering.

In recent years, NLP has grown tremendously due to the availability of large datasets and powerful computing resources. Applications of NLP can be seen in everyday life, from chatbots and virtual assistants to language translators and recommendation systems. Companies like Google, Microsoft, Amazon, and Apple invest heavily in NLP research to enhance user experiences.

Preprocessing is one of the most critical steps in any NLP pipeline. This includes cleaning the text data by removing unnecessary symbols, punctuation, and whitespace. Tokenization is used to split the text into words or sentences. Further, stopword removal eliminates common but less meaningful words like "the", "is", "and". Lemmatization and stemming are used to normalize the words to their base or root form.

Another important step is POS (Part-of-Speech) tagging, where each word in the text is marked with its respective part of speech, such as noun, verb, adjective, etc. POS tagging helps in better understanding the grammatical structure of the sentence and can be useful in many NLP tasks like named entity recognition, dependency parsing, and syntactic analysis.
"""

print(" Original Corpus:")
print(corpus)

tokens = word_tokenize(corpus)
print("\n Tokenized Words (first 40):")
print(tokens[:40])

tokens_no_punct = [word for word in tokens if word not in string.punctuation]
print("\n After Removing Punctuation (first 40):")
print(tokens_no_punct[:40])

tokens_no_whitespace = [word.strip() for word in tokens_no_punct if word.strip() != '']
print("\n After Removing Extra Whitespaces (first 40):")
print(tokens_no_whitespace[:40])

stop_words = set(stopwords.words('english'))
tokens_no_stopwords = [word for word in tokens_no_whitespace if word.lower() not in stop_words]
print("\n After Stopword Removal (first 40):")
print(tokens_no_stopwords[:40])

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

porter_stemmed = [porter.stem(word) for word in tokens_no_stopwords]
lancaster_stemmed = [lancaster.stem(word) for word in tokens_no_stopwords]
snowball_stemmed = [snowball.stem(word) for word in tokens_no_stopwords]

print("\n After Porter Stemming (first 40):")
print(porter_stemmed[:40])

print("\n After Lancaster Stemming (first 40):")
print(lancaster_stemmed[:40])

print("\n After Snowball Stemming (first 40):")
print(snowball_stemmed[:40])

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
print("\n After Lemmatization (first 40):")
print(lemmatized[:40])

pos_tags = pos_tag(tokens_no_stopwords)
print("\n POS Tagging (first 40):")
print(pos_tags[:40])