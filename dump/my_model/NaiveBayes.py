import pandas as pd
import numpy as np
import spacy
import pickle
import string
import re
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

class Sentiment:
    def __init__ (self):
        self.nlp = spacy.load ("en_core_web_sm")
        self.neg_words = pickle.load (open("../neg_words.pkl", "rb"))
        self.stop_words = spacy.lang.en.STOP_WORDS

        # By default, not is a stopword
        if 'not' in self.stop_words:
            self.stop_words.remove ('not')
        self.stop_words.update (string.punctuation)
        self.split_pattern = re.compile (r"(\s|-)")

    def vectorizer (self, X):
        vect = TfidfVectorizer ()
        return vect.fit_transform (X)

    def train_model (self, X, y):
        gaussian_model = GaussianNB()
        multinomial_model = MultinomialNB()
        self.models = [gaussian_model, multinomial_model]
        X_train, X_test, y_train, y_test = train_test_split (X, y)

        for model in models:
            model.fit (X_train, y_train)
            predicted = model.predict (X_test)
            print (f"\t Confusion Matrix: {model}")
            print (confusion_matrix (y_test, predicted))

            print (f"\t Classification Matrix: {model}")
            print (classification_report (y_test, predicted))

            print ("------------ Over -----------------")
