# %%
import pandas as pd
import spacy
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from helper import plot_and_add_hi_freq_feature, plot_frequency_distribution_of_ngrams, make_feature, get_unique
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Load Spacy large
nlp = spacy.load("en_core_web_lg")

# %%


def wrangle(X_train, X_test, features, tfidf_def=None, SVD_def=None, include_text=True, one_hot=True):
    lsa = get_pipe('combined', tfidf_def, SVD_def)
    n1, n2, n3 = len(X_train), len(X_test)
    X = pd.concat([X_train, X_test], ignore_index=True)
    if include_text:
        dtm = lsa.fit_transform(X['blurb'])
        dtm = pd.DataFrame(data=dtm, index=X.index)
    else:
        dtm = pd.DataFrame()
    ratings = ['successful', 'failed']
    if not one_hot:
        for rating in ratings:
            for feature in features:
                col_name = rating + "_" + feature
                dtm[col_name] = X[col_name]
                dtm[col_name] = X[col_name]
                dtm[col_name] = X[col_name]
    else:
        ohe = ohe_cols(X)
        dtm = pd.concat([dtm, ohe], axis=1)
    dtm_training   = dtm.iloc[0:n1]
    dtm_test       = dtm.iloc[n1:n1+n2]

    return dtm_training, dtm_test


# %%
target    = 'state'
df_train = pd.read_pickle('./df_train.pkl')
df_test  = pd.read_pickle('./df_test.pkl')

X_train, y_train = df_train.drop(columns=[target]), df_train[target]
X_test, y_test  = df_test.drop(columns=[target]), df_test[target]

# %%
nlp_cols  = ['blurb', 'name']
cat_cols  = ['country', 'spotlight', 'staff_pick', 'currency']
num_cols  = ['converted_pledged_amount', 'backers_count', 'goal']
target    = 'state'
date_cols = ['launched_at', 'deadline']

# %%
feature , figsize, num_ngrams, ngram_range = 'blurb', (20,20), 50, (1,2)

X_train, X_test = plot_and_add_hi_freq_feature(X_train, y_train, X_test, feature, ngram_range, num_ngrams, figsize)

# %%


# %%
