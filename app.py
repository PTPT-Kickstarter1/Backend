# %%
from sklearn.metrics import accuracy_score
import pandas as pd
import spacy
import category_encoders as ce
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from helper import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
import streamlit as st
nlp       = spacy.load("en_core_web_lg")

def my_ks_blurb_and_gram_count(X,tfidf_def=None, SVD_def=None, include_text=True):
    """ 
    Pre-process and Create Document Term Matrix for "blurb" feature via pipe
    
    Args:
        X_train_in (DataFrame): X Training Set in, after plot_and_add_hi_freq feature is run
        X_test_in (DataFrame: X Test Set in, after plot_and_add_hi_freq feature is run
        tfidf_def (Dict, optional): Input Args to Initialize TfidfVectorizer. Defaults to None.
        SVD_def (Dict, optional): nput Args to Initialize SVD. Defaults to None.
        include_text (bool, optional): . Defaults to True.

    Returns:
        dtm_train (DataFrame), dtm_test(DataFrame) : Document Term Matrix
    """
    
    # Get LSA Pre-Process Pipeline (no classifier added yet):
    lsa    = get_pipe('combined', tfidf_def, SVD_def)
    
    
    # If True create DTM with lsa.fit_transform on 'blurb' column
    # If False, Not implemented yet
    if include_text:
        dtm = lsa.fit_transform(X['blurb'])
        dtm = pd.DataFrame(data=dtm, index=X.index)
        
    else:
    # Functionality for non-default case not implemented yet
        return NotImplementedError()
        dtm = pd.DataFrame()
    
    # One Hot Encode Our Added Feature (gram_count_cols)
    ohe            = ohe_gram_count_cols(X)
    
    # Concat blurb Document Term Matrix and One Hot Encoded gram count cols:
    dtm            = pd.concat([dtm, ohe], axis=1, ignore_index=True)


    return dtm

def numeric_transform(X):
        
    # Impute missing numerical features using median and normalize:
    pipe = make_pipeline(
        SimpleImputer(strategy = 'median'),
        Normalizer(norm='l1'),
    )

    # Fit Transform on Combined Train/Test Set:
    dtm_out = pipe.fit_transform(X)
    
    return dtm_out


def load(my_ks):
    # Use NLP cols to create new feature counting most occurence of most common but unique to successful/failed campaigns
    # In model, NLP col is TfidfVectorized and then run through SVD
    nlp_cols  = ['blurb']

    # Categorical features used in model:
    cat_cols  = ['country', 'spotlight', 'staff_pick', 'currency']

    # Numerical features used in model:
    num_cols  = ['backers_count', 'goal', 'campaign_length']
 

    model_cols = num_cols + cat_cols + nlp_cols

    # Load training and test data frames in from pickle:
    target    = 'state'
    df_train  = pd.read_pickle('./df_train.pkl')
    df_test   = pd.read_pickle('./df_test.pkl')


    #  Seperate Train/Test into X / y :
    X_train, y_train = df_train.drop(columns=[target]), df_train[target]
    X_test, y_test   = df_test.drop(columns=[target]), df_test[target]
    # Remove columns not used by model so we can add out kick starter
    X_train = X_train[model_cols]
    X_test = X_test[model_cols].reset_index()
    
    X_test = pd.concat([X_test, my_ks])


    # Remove columns not used by model so we can add out kick starter
    X_train = X_train[model_cols]
    X_test = X_test[model_cols]
    X_train_numeric = X_train[num_cols]
    X_test_numeric  = X_test[num_cols]

    # Concat into one matrice so all feature engineering is performed uniformly on train/test sets:
    X_combined_numeric = pd.concat((X_train_numeric, X_test_numeric), ignore_index=True)
    dtm_train_test_num = numeric_transform(X_combined_numeric)
    # Seperate back out into train/test set:
    n1, n2        = len(X_train), len(X_test)
    dtm_train_num = dtm_train_test_num[0:n1]
    dtm_test_num  = dtm_train_test_num[n1:n1+n2]


    # 1. Get the <num_ngrams> most common <ngram_range> words/ phrases in the description of successful and failed campaigns
    # 2. Plot the frequency distirbution for successful/failed phrases
    # 3. Find phrases unique to successful or failed campaigns 
    # 4. Count number of occurences of most common unique successful and failed phrases in each project's description
    # 5. Create feature of count for successful and failed
    # 6. One Hot Encode Created Feature
    feature, figsize, num_ngrams, ngram_range = 'blurb', (20,20), 50, (1,2)

    X_train, X_test, vect, success, fail = plot_and_add_hi_freq_feature(X_train, y_train, X_test, feature, ngram_range, num_ngrams, figsize)

    # Set Parameters for TfidfVectorizer:
    tfidf_def = {
        'max_df': .7,
        'min_df': .2,
        'max_features': 30,
    }

    # Set Parameters for Singular Value Decomposition
    SVD_def = {
        'n_components': 5,
        'algorithm': 'randomized',
        'n_iter': 100
        }

    # Run 
    dtm_train, dtm_test = blurb_and_gram_count(X_train, X_test, tfidf_def, SVD_def, True)
    # Combine dtm from Tdidf/SVD of blurb
    dtm_train = np.concatenate([dtm_train, dtm_train_num], axis=1)
    dtm_test  = np.concatenate([dtm_test, dtm_test_num], axis=1)

    
    return vect, success, fail, dtm_train, y_train, dtm_test, y_test, X_test

def train(dtm_train, y_train, dtm_test, y_test):
    n =  len(dtm_test)
    # Initialize RandomForestClassifier with default setting and fit model on test
    clf = RandomForestClassifier()
    clf = clf.fit(dtm_train, y_train)
    y_pred = clf.predict(dtm_test)
    score  = accuracy_score(y_pred[0:n-2], y_test[0:n-2])
    
    my_ks_pred = y_pred[n-1]
    return my_ks_pred, score


def nn(X_test, dtm_test):
    nn = NearestNeighbors(n_neighbors=6, algorithm="kd_tree")
    # Fit on DTM
    nn.fit(dtm_test)

    # Pull our added ideal description to query listing
    doc_vector = [dtm_test[-1]]

    # Query Using kneighbors 
    neigh_dist, neigh_ind = nn.kneighbors(doc_vector)

    neigh_ind = neigh_ind.flatten()
    nearest = X_test['blurb'].iloc[neigh_ind[1:]]
    return nearest


country_options  = ['US', 'GB', 'NL', 'CA', 'PL', 'IT', 'HK', 'MX', 'NO', 'DE', 'DK', 'ES', 'CH', 'AU', 'FR', 'NZ', 'SG', 'BE', 'IE', 'SE', 'AT']
currency_options = ['USD', 'GBP', 'EUR', 'CAD', 'HKD', 'MXN', 'NOK', 'DKK', 'CHF', 'AUD', 'NZD', 'SGD', 'SEK']
with st.form(key='my_kickstarter'):
    ks_idea         = st.text_input(label="Enter Kickstarter Idea")
    num_backers     = st.slider(label="Select # of Backers You Believe You Can Get", min_value=0, max_value= 1000, step=10, format='%i')
    campaign_length = st.slider(label="Campaign Length:", min_value=0, max_value= 100, step=5)
    country         = st.selectbox(label="Select Country", options=country_options)
    currency        = st.selectbox(label="Select Currency", options=currency_options)
    staff_pick      = st.radio(label="Staff Pick?", options=[True,False])
    spotlight       = st.radio(label="Spotlight?",options=[True,False])
    goal            = st.slider(label = 'Select Goal:', min_value=1, max_value=10**5)
    submit          = st.form_submit_button(label="Will My Kickstarter Succeed?")
    if submit:
        ks_test = {'backers_count'  : [num_backers],
        'goal'           : [goal],
        'campaign_length': [campaign_length],
        'country'        : [country],
        'spotlight'      : [spotlight],
        'staff_pick'     : [staff_pick],
        'currency'       : [currency],
        'blurb'          : [ks_idea]
        }
        my_ks = pd.DataFrame.from_dict(ks_test)
        st.write(my_ks)
        
        vect, success, fail, dtm_train, y_train, dtm_test, y_test, X_test = load(my_ks)
        my_ks_pred, score = train(dtm_train, y_train, dtm_test, y_test)
        # nearest = nn(X_test, dtm_test)
        # st.write("Nearest Neighbors:")
        # st.write(nearest.iloc[1])
        # st.write(nearest.iloc[2])
        # st.write(nearest.iloc[3])
        

        # nlp_cols  = ['blurb']
        
        #     # Categorical features used in model:
        # cat_cols  = ['country', 'spotlight', 'staff_pick', 'currency']

        #     # Numerical features used in model:
        # num_cols  = ['backers_count', 'goal', 'campaign_length']

        # my_ks = make_feature(my_ks, vect, success, fail, 'blurb')
        # my_ks_num = my_ks[num_cols]
        # dtm_my_ks_num = numeric_transform(my_ks_num)
        
            
        #     # Set Parameters for TfidfVectorizer:
        # tfidf_def = {
        #     'max_df': .7,
        #     'min_df': .2,
        #     'max_features': 30,
        # }

        # # Set Parameters for Singular Value Decomposition
        # SVD_def = {
        #     'n_components': 5,
        #     'algorithm': 'randomized',
        #     'n_iter': 100
        #     }
        # dtm_my_ks     = my_ks_blurb_and_gram_count(my_ks, tfidf_def, SVD_def)
        # my_dtm   = pd.concat([dtm_my_ks, dtm_my_ks_num], axis =1)

        st.write(my_ks_pred)

            
            

    
# %%
