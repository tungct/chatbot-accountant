# https://github.com/parulnith/Building-a-Simple-Chatbot-in-Python-using-NLTK

# similar: https://medium.com/swlh/a-chatbot-in-python-using-nltk-938a37a9eacc

# Meet Robo: your friend

# import necessary libraries
import io
import random
import string  # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('popular', quiet=True)  # for downloading packages

# uncomment the following only the first time
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only


# Reading in the corpus
with open('vi-chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# TOkenisation
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

TfidfVec = TfidfVectorizer(tokenizer=LemNormalize)
tfidf = TfidfVec.fit_transform(sent_tokens)

new_doc = ['tôi muốn xem bảng tồn kho thì phải nhập thế nào vào phần mềm']

new_vec = TfidfVec.transform(new_doc).toarray()
print(new_vec)
vals = cosine_similarity(tfidf[0], new_vec)
print(vals)



