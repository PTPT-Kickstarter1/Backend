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
from sklearn.neighbors import NearestNeighbors


# %%
# Load Spacy Large:
nlp       = spacy.load("en_core_web_lg")

# Load training and test data frames in from pickle:
target    = 'state'
df_train  = pd.read_pickle('./df_train.pkl')
df_test   = pd.read_pickle('./df_test.pkl')

#  Seperate Train/Test into X / y :
X_train, y_train = df_train.drop(columns=[target]), df_train[target]
X_test, y_test   = df_test.drop(columns=[target]), df_test[target]

# %%
# Use NLP cols to create new feature counting most occurence of most common but unique to successful/failed campaigns
# In model, NLP col is TfidfVectorized and then run through SVD
nlp_cols  = ['blurb']

# Categorical features used in model:
cat_cols  = ['country', 'spotlight', 'staff_pick', 'currency']

# Numerical features used in model:
num_cols  = ['backers_count', 'goal', 'campaign_length']

# %% 
# Select numerical features from training/test set
X_train_numeric = X_train[num_cols]
X_test_numeric  = X_test[num_cols]

# Concat into one matrice so all feature engineering is performed uniformly on train/test sets:
X_combined_numeric = pd.concat((X_train_numeric, X_test_numeric), ignore_index=True)

# Impute missing numerical features using median and normalize:
pipe = make_pipeline(
    SimpleImputer(strategy = 'median'),
    Normalizer(norm='l1'),
)

# Fit Transform on Combined Train/Test Set:
dtm_train_test_num = pipe.fit_transform(X_combined_numeric)

# Seperate back out into train/test set:
n1, n2        = len(X_train), len(X_test)
dtm_train_num = dtm_train_test_num[0:n1]
dtm_test_num  = dtm_train_test_num[n1:n1+n2]

# %%
# 1. Get the <num_ngrams> most common <ngram_range> words/ phrases in the description of successful and failed campaigns
# 2. Plot the frequency distirbution for successful/failed phrases
# 3. Find phrases unique to successful or failed campaigns 
# 4. Count number of occurences of most common unique successful and failed phrases in each project's description
# 5. Create feature of count for successful and failed
# 6. One Hot Encode Created Feature
feature , figsize, num_ngrams, ngram_range = 'blurb', (20,20), 50, (1,2)

X_train, X_test = plot_and_add_hi_freq_feature(X_train, y_train, X_test, feature, ngram_range, num_ngrams, figsize)
# %%
# Set Parameters for TfidfVectorizer:
tfidf_def = {
    'max_df': .7,
    'min_df': .2,
    'max_features': 300,
}

# Set Parameters for Singular Value Decomposition
SVD_def = {
    'n_components': 5,
    'algorithm': 'randomized',
    'n_iter': 100
    }

# Run 
dtm_train, dtm_test = blurb_and_gram_count(X_train, X_test, tfidf_def, SVD_def, True)
# %%
# Combine dtm from Tdidf/SVD of blurb
dtm_train = np.concatenate([dtm_train, dtm_train_num], axis=1)
dtm_test  = np.concatenate([dtm_test, dtm_test_num], axis=1)

# %%
# Initialize RandomForestClassifier with default setting and fit model on test
clf                 = RandomForestClassifier()
clf                 = clf.fit(dtm_train, y_train)

# %%
# Predict target for test data and calculate accuracy:
y_pred = clf.predict(dtm_test)
score  = accuracy_score(y_pred, y_test)
print(score)

# %% INWORK
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, algorithm="kd_tree")

# test description:
ks_description = "An App that helps you select a kickstarter project likely to succeed"

# add ideal description to end of job_listing DataFrame (copy and create new df)
job_listing_w_ideal = job_listing.copy()
job_listing_w_ideal.loc[len(job_listing.index)] = [ideal_description, 'Data Scientist']

# vectorize job_listing w/ ideal description added using Tfidf 
# this step is required if our ideal posting introduces any new vocabulary
wc_ideal, dtm_ideal = vec(job_listing_w_ideal, 'TfidfVectorizer')

# Fit on DTM
nn.fit(dtm_ideal)

# Pull our added ideal description to query listing
doc_vector = [dtm_ideal.iloc[len(dtm_ideal)-1]]

# Query Using kneighbors 
neigh_dist, neigh_ind = nn.kneighbors(doc_vector)