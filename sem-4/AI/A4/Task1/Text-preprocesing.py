# Question 1: Text Preprocessing and Feature Extraction for NLP

# For Task
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
import spacy

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
print(documents[0], "\n")
print(documents[1], "\n")


# Apply Text Preprocessing Techniques

# Setup:

# Timing wrapper:
def take_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

# Sample document for initial comparison:
sample = documents[0]

# NLTK Setup:
@take_time
def NLTK_Tokenize(text):
    return nltk.word_tokenize(text)

@take_time
def NLTK_Stemming(NLTK_tokens):    
    p_stemmer = nltk.stem.porter.PorterStemmer()
    stemmed_tokens = [
        [
            p_stemmer.stem(token) for token in tokens
        ] for tokens in NLTK_tokens
    ]
    return stemmed_tokens


# SpaCy Setup:
print("Setting up SpaCy...")
spacy.prefer_gpu()  # Enable GPU if available.
# Disabeling useless fields (for higher speed).
nlp_token_only = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"])
nlp_lemmatize = spacy.load("en_core_web_sm", disable=["parser", "ner"])

@take_time
def SpaCy_Tokenize(text, nlp):
    if isinstance(text, str): # For the sample document
        doc = nlp(text)
        return [token.text for token in doc]
    else:
        tokens = []
        for doc in nlp.pipe(text, batch_size=300):
            batch = [token.text for token in doc]
            tokens.append(batch)
        return tokens

@take_time
def SpaCy_Lemmatize(token_list, nlp):
    lemmatized_tokens = []
    # Joining the list of tokens into strings for nlp.pipe:
    joined_text = [' '.join(tokens) for tokens in token_list]
    for doc in nlp.pipe(joined_text, batch_size=300):
        tokens = [token.lemma_ for token in doc]
        lemmatized_tokens.append(tokens)
    return lemmatized_tokens


# Initial Comparison:
print("Comparing NLTK and SpaCy on sample document...")
NLTK_result, NLTK_duration = NLTK_Tokenize(sample)
SpaCy_result, SpaCy_duration = SpaCy_Tokenize(sample, nlp_token_only)

print(f"NLTK result: \n{NLTK_result}\n")
print(f"SpaCy result: \n{SpaCy_result}\n")
print(f"NLTK time: {NLTK_duration:.4f} seconds")
print(f"SpaCy time: {SpaCy_duration:.4f} seconds\n")


# NLTK Processing (All Documents):
print("NLTK Tokenizing all documents...")
NLTK_tokens, NLTK_total_duration = [], 0
for doc in documents:
    tokens, duration = NLTK_Tokenize(doc)
    NLTK_tokens.append(tokens)
    NLTK_total_duration += duration

stop_words = set(nltk.corpus.stopwords.words("english"))

print("NLTK Filtering out stopwords and punctuation...")
filtered_NLTK_tokens = [
    [
        token for token in tokens 
        if token not in stop_words and token not in string.punctuation
    ] for tokens in NLTK_tokens
]

# NLTK Stemming:
print("NLTK Stemming...")
stemmed_tokens, stemming_duration = NLTK_Stemming(filtered_NLTK_tokens)

# SpaCy Processing (All Documents):
print("SpaCy Tokenizing all documents...")
SpaCy_tokens, SpaCy_total_duration = SpaCy_Tokenize(documents, nlp_token_only)

print("SpaCy Filtering out punctuation...")
SpaCy_tokens = [
    [
        token for token in tokens if token not in string.punctuation
    ] for tokens in SpaCy_tokens    
]

# SpaCy Lemmatization:
print("SpaCy Lemmatizing...")
lemmatized_tokens, lemmatize_duration = SpaCy_Lemmatize(SpaCy_tokens, nlp_lemmatize)


'''
Feature Extraction using TF-IDF
'''

print("Computing TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer()

print("- Parsing tokens...")
without_stopwords = [
    [
        token for token in tokens if token not in stop_words
    ] for tokens in lemmatized_tokens
]

# Join tokens back into strings for TF-IDF:
texts_raw = [' '.join(tokens) for tokens in lemmatized_tokens]
texts_without_stopwords = [' '.join(tokens) for tokens in without_stopwords]

# Computing TF-IDF features:
tfidf_raw = tfidf_vectorizer.fit_transform(texts_raw)
tfidf_stopwords = tfidf_vectorizer.fit_transform(texts_without_stopwords)

print("\nAnalyzing top 5 words per category based on TF-IDF scores (without stopwords)...")
feature_names = tfidf_vectorizer.get_feature_names_out()
top_n = 5
for category_idx, category_name in enumerate(category_names):
    category_docs = np.where(categories == category_idx)[0]
    if len(category_docs) == 0:
        continue
    category_tfidf = tfidf_stopwords[category_docs].mean(axis=0).A1
    top_indices = category_tfidf.argsort()[-top_n:][::-1]
    top_words = [(feature_names[idx], category_tfidf[idx]) for idx in top_indices]
    print(f"\nCategory: {category_name}")
    print(f"Top {top_n} words (TF-IDF score):")
    for word, score in top_words:
        print(f"  {word}: {score:.4f}")

'''
Training a simple Classifier:
'''

def evaluate_model(tfidf_matrix, labels, target_names, description):
    print(f"\n===== Evaluation: {description} =====")
    
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, labels, test_size=0.2, random_state=42
    )

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=target_names))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

accuracy_raw = evaluate_model(tfidf_raw, categories, category_names, "Raw tokens.")
accuracy_stopwords = evaluate_model(tfidf_stopwords, categories, category_names, "Without stopwords")

# Results:
print("\nPreprocessing Results:")
print(f"NLTK Total Tokenization Time: {NLTK_total_duration:.4f} seconds")
print(f"SpaCy Total Tokenization Time: {SpaCy_total_duration:.4f} seconds")
print(f"NLTK Stemming Time: {stemming_duration:.4f} seconds")
print(f"SpaCy Lemmatization Time: {lemmatize_duration:.4f} seconds")
print(f"Sample Filtered NLTK Tokens (first document, first 5): {filtered_NLTK_tokens[0][:5]}")
print(f"Sample Filtered SpaCy Tokens (first document, first 5): {SpaCy_tokens[0][:5]}")
print(f"Sample Stemmed Tokens (first document, first 5): {stemmed_tokens[0][:5]}")
print(f"Sample Lemmatized Tokens (first document, first 5): {lemmatized_tokens[0][:5]}")
print(f"\nAccuracy with stopwords: {accuracy_raw}")
print(f"Accuracy without stopwords: {accuracy_stopwords}")
