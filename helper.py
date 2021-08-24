from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from collections import Counter
import category_encoders as ce
import spacy
from sklearn.model_selection import train_test_split
import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt


def blurb_and_gram_count(X_train_in, X_test_in, tfidf_def=None, SVD_def=None, include_text=True):
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

    # Copy input dataframes in and get lengths:
    X_train, X_test = X_train_in.copy(), X_test_in.copy()
    n1, n2          = len(X_train), len(X_test)
    
    # Get LSA Pre-Process Pipeline (no classifier added yet):
    lsa    = get_pipe('combined', tfidf_def, SVD_def)
    
    # Combine Train/Test Set (Ensure Pre-Process/Wranglinging Performed Uniformly):
    X      = pd.concat([X_train, X_test], ignore_index=True)
    
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
    
    # Split back into Training / Test Set:
    dtm_training   = dtm.iloc[0:n1]
    dtm_test       = dtm.iloc[n1:n1+n2]

    return dtm_training, dtm_test


def ohe_gram_count_cols(X_in):
    """ One Hot Encode gram_count_cols.

    Args:
        X_in (DataFrame): Combined Training/Test DataFrame w/ added features.

    Returns:
        ohe [DataFrame]: Combined Training/Test Dataframe After One Hot Encode Transform to gram_count_cols
    """
    # Cooy X_in
    X_out    = X_in.copy()
    columns  = X_in.columns
    ohe_cols = []
    
    # There is probably a better way to do the below.
    # Category Encoders can't One Hot Encode (int) features, so map integers to characters 
    
    # Define Mapping from integer to characters
    alphabet    = 'abcdefghijklmnopqrstuvwxyz'
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    
    # Our added columns have prefix "succesful_" or "failed_", select these columns:
    for col in columns:
        if ('successful' in col) or ('failed' in col) :
            ohe_cols.append(col)

    # Select Features to One Hot Encode:
    ohe_df     = X_out[ohe_cols].copy()
    # Initialize OutputDataFrame
    ohe_df_out = pd.DataFrame(columns=ohe_df.columns)
    
    # Remap integers in features to characters:
    for col, data in ohe_df.items():
        for index, value in data.items():
            ohe_df_out.at[index, col] = int_to_char[value]

    # One Hot Encode Remapped Features
    enc = ce.OneHotEncoder(cols=ohe_cols, use_cat_names=False)
    ohe = enc.fit_transform(ohe_df_out)
    
    # Return just the one hot encoded columns
    return ohe


def get_pipe(pipe_type, tfidf_def=None, SVD_def=None):
    """
    Set-Up Pipeline with Simple Full Model or Pre-procesing steps for use on subset of input data.
    
    :param pipe_type (string): lsa, svd, or combined:
        lsa      - Pipe with TfidfVectorizer, Truncated SVD, and Logistic Regression Classifier
        svd      - Pipe with TfidfVectorizer and LinearSVC
        comvined - Pipe with TfidfVectorizer and Truncated SVD (classifier step is added 
                        later on since we are running this on a subset of input features)
    :param tfidf_def (Dict): **kwargs to initialize TfidfVectorizer
    :param   SVD_def (Dict): ** kwargs to initialize TruncatedSVD
    
    :return: pipe_out (sklearn.Pipeline): Selected Pipeline
    """
    pipe_out           = []
    
    # Get **kwargs for tfidf/SVD initialization
    tfidf_def, SVD_def = get_tfidf_svd_def(tfidf_def, SVD_def)
    vect               = TfidfVectorizer(**tfidf_def)
    svd                = TruncatedSVD(**SVD_def)
    
    svc = LinearSVC()
    lr  = LogisticRegression(solver='lbfgs')
    lsa = Pipeline([('vect', vect), ('svd', svd)])

    if pipe_type == 'lsa':
        pipe_out = Pipeline([
            ('lsa', lsa),
            ('clf', lr)
        ])

    elif pipe_type == 'svc':
        pipe_out = Pipeline([
            ('vect', vect),
            ('clf', svc)]
        )

    elif pipe_type == 'combined':
        pipe_out = lsa

    return pipe_out


def get_tfidf_svd_def(tfidf_def, SVD_def):
    """ Gets Default **kwawargs for Truncated SVD and TfidfVectorizer used in model.

    Args:
        tfidf_def (Dict): Dict with **kwargs for TfidfVectorizer initialization
        SVD_def (Dict): Dict with **kwargs for Truncated SVD initialization
    """
    if tfidf_def is None:
        tfidf_def = {'stop_words': 'english',
                    'ngram_range': (1, 2),
                    'min_df'     : 2,
                    'max_df'     : .5
                    }
        
    if SVD_def is None:
        SVD_def = {'n_components': 20,
                   'algorithm'   : 'randomized',
                   'n_iter'      : 100
                   }
    
    return tfidf_def, SVD_def


def make_feature(X_in, vect, success, fail, feature):
    """ Make <state>_ngram_counts features. Feature counts occurence of 
    words/phrases most frequent and unique to each target state
    (<state>: successful or failed)

    Args:
        X_in (DataFrame)      : Combined X_train / X_test DataFrame in
        vect (CountVectorizer): Initialized CountVectorizer
        success (list)        : List of words/phrases uniquely and most frequently appearing in successful <feature>
        fail (list)           : List of words/phrases uniquely and most frequently appearing in failed <feature>
        feature (string))     : Existing Text Feature Column Used to Create New Feature

    Returns:
        X(DataFrame): Input DataFrame with Added Features
    """
    X = X_in.copy()
    feature_txt = X[feature].tolist()
    vect.fit(feature_txt)
    unique_grams = success + fail

    dtm  = vect.transform(feature_txt)
    df   = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())

    df   = df[df.columns.intersection(unique_grams)]
    
    s_list, f_list = [], []
    for i in range(0, len(df)):
        row = df.iloc[i]
        s_count, f_count = 0, 0
        
        for gram in unique_grams:
            count_at_ngram = row.loc[gram]
            
            if count_at_ngram > 0:
                if gram in success:
                    s_count += 1
                elif gram in fail:
                    f_count += 1

        s_list.append(s_count)
        f_list.append(f_count)
        col1 = 'successful' + "_" + feature
        col2 = 'failed'     + "_" + feature

    X[col1] = s_list
    X[col2] = f_list
    return X


def get_unique(text_in, feature='blurb'):
    """ Get Most Common words/phrases in successful/failed 
        campaign for text <feature> that do not appear in the other class.
        
    Args:
        text_in (Dict): Dictionary with most common words/phrases for <feature> associated with each target class
        feature (str, optional): [description]. Defaults to 'blurb'.

    Returns:
        success_out (list): Most common words/phrases associated with text <feature> unique to successful target class
        fail_out    (list): Most common words/phrases associated with text <feature> unique to successful target class
    """
    success_out, fail_out = [], []
    target_class = ['successful', 'failed']
    # Dict Keys for <feature> of word count for each target class
    key1    = target_class[0] + "_ngrams_" + feature
    key2    = target_class[1] + "_ngrams_" + feature
    
    # Most common words/phrases in each target class' text <feature>
    success = set(text_in[key1])
    fail    = set(text_in[key2])
    
    # Get words/phrases unique to each target class
    for item in success:
        if (item not in fail) :
            success_out.append(item)
    for item in fail:
        if (item not in success) :
            fail_out.append(item)

    return success_out, fail_out


def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range = (1, 2),
                                          num_ngrams = 50,
                                          title = 'Frequency distribution of n-grams',
                                          figsize = (20, 20) 
                                          ):
    """Plots the frequency distribution of n-grams.

        # Arguments
            samples_texts: list, sample texts.
            ngram_range: tuple (min, mplt), The range of n-gram values to consider.
                Min and mplt are the lower and upper bound values for the range.
            num_ngrams: int, number of n-grams to plot.
                Top `num_ngrams` frequent n-grams will be plotted.
        """
    # Create args required for vectorizing.
    kwargs = {
        'ngram_range'  : ngram_range,
        'dtype'        : 'int32',
        'stop_words'   : 'english',
        'strip_accents': 'unicode',
        'decode_error' : 'replace',
        'analyzer'     : 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.figure(figsize=figsize)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title(title)
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()
    return ngrams, counts


def my_tokenizer(text):
    clean_text = re.sub('[^a-zA-Z ]', '', text)
    tokens = clean_text.lower().split()
    return tokens


def plot_and_add_hi_freq_feature(X_train_in, 
                                 y_train_in, 
                                 X_test_in,
                                 feature, 
                                 ngram_range=(1,2),
                                 num_ngrams=50,
                                 figsize=(20,20)):
    """ Plot Frequency Distribution for top <num_ngrams> in <feature> for each class of target

    Args:
        X_train_in (DataFrame): X Train In
        y_train_in (DataFrame): y Train In
        X_test_in (DataFrame): X Test in
        feature (string): Column Name of feature to be plotted and used to create new feature
        ngram_range (tuple, optional): range of number of words to be considered for each ngram. Defaults to (1,2).
        num_ngrams (int, optional): # of ngrams to plot. Defaults to 50.
        figsize (tuple, optional): Figure Size. Defaults to (20,20).

    Returns:
        [type]: [description]
    """
    # Copy input dataframes
    X_train, y_train, X_test = X_train_in.copy(), y_train_in.copy(), X_test_in.copy()
    target_class             = ['successful', 'failed']
    
    # Initialize Dict with unique and most frequent word/phrases for each target class
    # and their counts
    blurb = {}
    for state in target_class:
        # Select subset of input dataframe associated with each target class:
        idx          = (y_train == state)
        X_state      = X_train[idx]
        
        # Select Text Feature, Conver
        state_text   = X_state[feature].tolist()
        title        = 'Frequency Distribution of ngrams for ' + state
        
        ngrams, counts = plot_frequency_distribution_of_ngrams(state_text,
                                                            ngram_range,
                                                            num_ngrams,
                                                            title,
                                                            figsize)
        key1        = state + "_ngrams_" + feature
        key2        = state + "_counts_" + feature
        blurb[key1] = ngrams
        blurb[key2] = counts

    # Get the unique ngrams with the highest frequency associated with each class and create numerical feature
    # with how often the highest frequency, unique ngram appeared in the description
    success, fail = get_unique(blurb, feature)
    kwargs = {
        'ngram_range'  : ngram_range,
        'dtype'        : 'int32',
        'stop_words'   : 'english',
        'strip_accents': 'unicode',
        'decode_error' : 'replace',
        'analyzer'     : 'word',  # Split text into word tokens.
    }
    vect         = CountVectorizer(**kwargs)
    unique_grams = success + fail
    n1, n2       = len(X_train), len(X_test_in)
    X_combined   = pd.concat([X_train, X_test],ignore_index=True)
    X_combined   = make_feature(X_combined, vect, success, fail, feature)
    
    X_train = X_combined.iloc[0:n1]
    X_test  = X_combined.iloc[n1:n1 + n2]

    return X_train, X_test