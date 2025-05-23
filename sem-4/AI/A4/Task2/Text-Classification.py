# Question 2:

# For Task
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Helper
from functools import wraps
import string
import time


# Loading dataset:
print("Fetching data...")
dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

print("Parsing data...")
documents = dataset.data
categories = dataset.target
category_names = dataset.target_names

# Lowercase all documents:
for i in range(len(documents)):
    documents[i] = documents[i].lower()

print("Printing sample documents...\n")
print(f"--Label: {category_names[0]}, data:--\n{documents[0]}\n")
print(f"--Label: {category_names[1]}, data:--\n{documents[1]}\n")


# Apply Text Preprocessing Techniques:

# Timing wrapper:
def take_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

# NLTK Setup:
@take_time
def NLTK_Tokenize(text):
    return nltk.word_tokenize(text)


# NLTK Processing (All Documents):
print("NLTK Tokenizing all documents...")
NLTK_tokens, NLTK_total_duration = [], 0
for doc in documents:
    tokens, duration = NLTK_Tokenize(doc)
    NLTK_tokens.append(tokens)
    NLTK_total_duration += duration

print(f"Time elapsed: {NLTK_total_duration}")

stop_words = set(nltk.corpus.stopwords.words("english"))

print("NLTK Filtering out stopwords and punctuation...")
filtered_NLTK_tokens = [
    [
        token for token in tokens 
        if token not in stop_words and token not in string.punctuation
    ] for tokens in NLTK_tokens
]



'''
Feature Extraction using TF-IDF
'''

print("Computing TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer()

print("Parsing tokens...")
texts = [' '.join(tokens) for tokens in filtered_NLTK_tokens] 
print("vectorizing...")
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

'''
Training a simple Classifier:
'''
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, categories, test_size=0.2, random_state=42
)

print("Training model...")
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=category_names))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(cm)

