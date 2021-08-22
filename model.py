# %%
from sklearn.metrics import accuracy_score
import pandas as pd
import spacy
import category_encoders as ce
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from helper import *
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# %%
def blurb_and_gram_count(X_train_in, X_test_in, tfidf_def=None, SVD_def=None, include_text=True):
    X_train, X_test = X_train_in.copy(), X_test_in.copy()
    lsa    = get_pipe('combined', tfidf_def, SVD_def)
    n1, n2 = len(X_train), len(X_test)
    X      = pd.concat([X_train, X_test], ignore_index=True)
    
    if include_text:
        dtm = lsa.fit_transform(X['blurb'])
        dtm = pd.DataFrame(data=dtm, index=X.index)
    else:
        dtm = pd.DataFrame()

    ohe            = ohe_gram_count_cols(X)
    dtm            = pd.concat([dtm, ohe], axis=1, ignore_index=True)
    dtm_training   = dtm.iloc[0:n1]
    dtm_test       = dtm.iloc[n1:n1+n2]

    return dtm_training, dtm_test
# %%
# Load Spacy large
nlp = spacy.load("en_core_web_lg")

target    = 'state'
df_train  = pd.read_pickle('./df_train.pkl')
df_test   = pd.read_pickle('./df_test.pkl')

X_train, y_train = df_train.drop(columns=[target]), df_train[target]
X_test, y_test   = df_test.drop(columns=[target]), df_test[target]

# %%
nlp_cols  = ['blurb']
cat_cols  = ['country', 'spotlight', 'staff_pick', 'currency']
num_cols  = ['converted_pledged_amount', 'backers_count', 'goal', 'pledge_pct_goal', 'campaign_length']

# %%
X_train_numeric = X_train[num_cols]
X_test_numeric  = X_test[num_cols]
X_combined_numeric = pd.concat((X_train_numeric, X_test_numeric), axis=1, ignore_index=True)

pipe = make_pipeline(
    SimpleImputer(strategy = 'median'),
    Normalizer(norm='l1'),
)

n1, n2 = len(X_train), len(X_test)
dtm_train_test_num = pipe.fit_transform(X_combined_numeric)
dtm_train_num = dtm_train_test_num[0:n1]
dtm_test_num  = dtm_train_test_num[n1:n1+n2]

# %%
feature , figsize, num_ngrams, ngram_range = 'blurb', (20,20), 50, (1,2)

X_train, X_test = plot_and_add_hi_freq_feature(X_train, y_train, X_test, feature, ngram_range, num_ngrams, figsize)
# %%
tfidf_def = {
    'max_df': .7,
    'min_df': .2,
    'max_features': 300,
}

SVD_def = {
    'n_components': 5,
    'algorithm': 'randomized',
    'n_iter': 100
    }

# One Hot Encode <state>_ngram_blurb cols and tfidf/SVD blurb 
dtm_train, dtm_test = blurb_and_gram_count(X_train, X_test, tfidf_def, SVD_def, True)
# %%
dtm_train = np.concatenate([dtm_train, dtm_train_num],axis=1)
dtm_test  = np.concatenate([dtm_test, dtm_test_num],axis=1)
# %%
clf                 = RandomForestClassifier()
clf                 = clf.fit(dtm_train, y_train)
# %%
y_pred = clf.predict(dtm_test)
score  = accuracy_score(y_pred, y_test)
# %%
