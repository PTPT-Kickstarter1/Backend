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
import spacy
from sklearn.model_selection import train_test_split
import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt

def ohe_cols(X_in):
    X_out = X_in.copy()
    columns = X_in.columns
    ohe_cols = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    # define a mapping of chars to integers
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    for col in columns:
        if ('successful' in col) or ('failed' in col) :
            ohe_cols.append(col)

    ohe_df    = X_out[ohe_cols].copy()
    ohe_df_out = pd.DataFrame(columns=ohe_df.columns)
    for col, data in ohe_df.items():
        for index, value in data.items():
            ohe_df_out.at[index, col] = int_to_char[value]

    enc = ce.OneHotEncoder(cols=ohe_cols, use_cat_names=False)
    ohe = enc.fit_transform(ohe_df_out)

    return ohe

def get_pipe(pipe_type, tfidf_def=None, SVD_def=None):
    """
    Set-Up Pipe
    :param pipe_type: lsa, svd, combined
    :param tfidf_def: **kwargs to initialize TfidfVectorizer
    :param SVD_def: ** kwargs to initialize TruncatedSVD
    :return:
    """
    pipe_out = []
    if tfidf_def is None:
        tfidf_def = {'stop_words': 'english',
                     'ngram_range': (1, 2),
                     'min_df': 2,
                     'max_df': .5
                     }
    if SVD_def is None:
        SVD_def = {'n_components': 20,
                   'algorithm': 'randomized',
                   'n_iter': 100
                   }

    vect = TfidfVectorizer(**tfidf_def)
    svm = LinearSVC()
    svd = TruncatedSVD(**SVD_def)
    lr = LogisticRegression(solver='lbfgs')
    lsa = Pipeline([('vect', vect), ('svd', svd)])

    if pipe_type == 'lsa':
        pipe_out = Pipeline([
            ('lsa', lsa),
            ('clf', lr)
        ])

    elif pipe_type == 'svd':
        pipe_out = Pipeline([
            ('vect', vect),
            ('clf', svm)]
        )

    elif pipe_type == 'combined':
        pipe_out = lsa

    return pipe_out

def make_feature(X, vect, unique_grams, success, fail, feature):
    
    feature_txt = X[feature].tolist()
    vect.fit(feature_txt)

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
    
    target_class = ['successful', 'failed']
    success_out, fail_out = [], []
    key1    = target_class[0] + "_ngrams_" + feature
    key2    = target_class[1] + "_ngrams_" + feature
    success = set(text_in[key1])
    fail    = set(text_in[key2])
    
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


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()
    
    def count(tokens):
        """
    Calculates some basic statistics about tokens in our corpus (i.e. corpus means collections text data)
    """
    # stores the count of each token

    word_counts = Counter()

    # stores the number of docs that each token appears in
    appears_in = Counter()
    total_docs = len(tokens)

    for token in tokens:
        # stores count of every appearance of a token
        word_counts.update(token)
        # use set() in order to not count duplicates, thereby count the num of docs that each token appears in
        appears_in.update(set(token))

    # build word count dataframe
    temp = zip(word_counts.keys(), word_counts.values())
    wc = pd.DataFrame(temp, columns=['word', 'count'])

    # rank the the word counts
    wc['rank'] = wc['count'].rank(method='first', ascending=False)
    total = wc['count'].sum()

    # calculate the percent total of each token
    wc['pct_total'] = wc['count'].apply(lambda token_count: token_count / total)

    # calculate the cumulative percent total of word counts
    wc = wc.sort_values(by='rank')
    wc['cul_pct_total'] = wc['pct_total'].cumsum()

    # create dataframe for document stats
    t2 = zip(appears_in.keys(), appears_in.values())
    ac = pd.DataFrame(t2, columns=['word', 'appears_in'])

    # merge word count stats with doc stats
    wc = ac.merge(wc, on='word')

    wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)

    return wc.sort_values(by='rank')


def my_tokenizer(text):
    clean_text = re.sub('[^a-zA-Z ]', '', text)
    tokens = clean_text.lower().split()
    return tokens


def plot_and_add_hi_freq_feature(X_train_in, 
                                 y_in, 
                                 X_test_in,
                                 feature, 
                                 ngram_range=(1,2),
                                 num_ngrams=50,
                                 figsize=(20,20)):
    X_train = X_train_in.copy()
    target_class = ['successful', 'failed']
    blurb = {}
    for state in target_class:
        idx          = (y_in == state)
        X_state      = X_train.loc[idx]
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
    n1, n2     = len(X_train), len(X_test_in)
    X_combined = pd.concat([X_train, X_test_in],ignore_index=True)
    X_combined = make_feature(X_combined, vect, unique_grams, success, fail, feature)
    
    X_train = X_combined.iloc[0:n1]
    X_test = X_combined.iloc[n1:n1 + n2]

    return X_train, X_test