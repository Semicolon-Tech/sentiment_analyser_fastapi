import string
from math import ceil

from tqdm import tqdm
import numpy as np
from sklearn.base import TransformerMixin

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

wordnet_lemma = WordNetLemmatizer()

nltk.download("punkt")
nltk.download("wordnet")

def clean_text(text: str):
    # removing upping case
    text = text.lower()

    # removing punctuation
    for char in string.punctuation:
        text = text.replace(char, " ")

    text = " ".join(([wordnet_lemma.lemmatize(word) for word in word_tokenize(text)]))
    return text


class DenseTransformer(TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self

    def transform(self, x, y=None, **fit_params):
        return 

    def __str__(self):
        return "DenseTransformer()"

    def __repr__(self):
        return self .__str__()


class CleanTextTransformer(TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return np.vectorize(clean_text)(x)

    def __str__(self):
        return "CleanTextTransformer()"

    def __repr__(self):
        return self .__str__()


cleanTextTransformer = CleanTextTransformer


class GaussianBatchNB(TransformerMixin):
    def __init__(self, batch_size, classes, *args, **kwargs):
        self._batch_size = batch_size
        self._classes = classes
        self._model = GaussianNB(*args, **kwargs)
        
    def fit(self, x, y, **fit_params):
        batch_size = self._batch_size
        
        for index in tqdm(range(batch_size, x.shape[0]+batch_size, batch_size)):
            self._model.partial_fit(
                x[index-batch_size:index, :].toarray(),
                y[index-batch_size:index], 
                classes=self._classes
            )                  
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return x
    
    def predict(self, x):
        batch_size = self._batch_size
        predictions = []
        for index in tqdm(range(batch_size, x.shape[0]+batch_size, batch_size)):
            predictions.append(
                self._model.predict(
                    x[index-batch_size:index, :].toarray()
                )
            )
        return np.array(predictions).flatten()
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)

    def __str__(self):
        return "GaussianBatchNB()"

    def __repr__(self):
        return self .__str__()