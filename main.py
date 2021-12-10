import shutup
shutup.please()
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from preprocessing import *
import os
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import pyLDAvis.gensim_models
import pyLDAvis
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from pprint import pprint
from multiprocessing import freeze_support

#Take tweets and convert to individual words.
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

#Create bigrams from tweets.
def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def main(num_topics):
    if(os.path.exists("cleaned_tweets.csv")):
        #Train LDA Model
        cleaned_tweets = pd.read_csv("cleaned_tweets.csv")
        #reals=cleaned_tweets[cleaned_tweets['target']==1]
        #reals=cleaned_tweets[cleaned_tweets['keyword']=="ablaze"]
        #ind=[len(str(x))>75 for x in cleaned_tweets["text"]]
        #reals=cleaned_tweets[ind]
        reals=cleaned_tweets
        data = reals['text'].values.tolist()
        data_words = list(sent_to_words(data))
        texts=data_words
        bigram_mod = bigrams(texts)
        bigram = [bigram_mod[review] for review in texts]
        id2word=corpora.Dictionary(bigram)
        id2word.filter_extremes(no_below=10, no_above=0.35)
        id2word.compactify()
        corpus=[id2word.doc2bow(text) for text in bigram]
        lda_model=gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
        doc_lda=lda_model[corpus]
        LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')

        # Extract LDA Model to Feature Vector
        training = []
        for i in range(len(reals)):
            best_topics = (lda_model.get_document_topics(corpus[i],minimum_probability=0.0))
            topic_vec = [best_topics[i][1] for i in range(num_topics)]
            training.append(topic_vec)

        # Train Classifier
        X = np.array(training)
        y = np.array(reals.target)

        kf = KFold(5, shuffle=True)
        lr_f1, lrs_f1, mhs_f1,  = [], [], []

        for train_ind, val_ind in kf.split(X, y):
            # Assign indices based on the kfolds data splitting.
            X_train, y_train = X[train_ind], y[train_ind]
            X_val, y_val = X[val_ind], y[val_ind]

            # Scale Data
            scaler = StandardScaler()
            scaled_X_tr = scaler.fit_transform(X_train)
            scaled_X_val = scaler.transform(X_val)
            # Logisitic Regression
            lr = LogisticRegression(
                class_weight= 'balanced',
                solver='newton-cg',
                fit_intercept=True
            ).fit(scaled_X_tr, y_train)
            # Logistic Regression SGD
            lrs = linear_model.SGDClassifier(
                max_iter=1000,
                tol=1e-3,
                loss='log',
                class_weight='balanced'
            ).fit(scaled_X_tr, y_train)
            # SGD Modified Huber
            mhs = linear_model.SGDClassifier(
                max_iter=1000,
                tol=1e-3,
                alpha=20,
                loss='modified_huber',
                class_weight='balanced'
            ).fit(scaled_X_tr, y_train)

            lr_f1.append(f1_score(y_val, lr.predict(scaled_X_val), average='binary'))
            lrs_f1.append(f1_score(y_val, lrs.predict(scaled_X_val), average='binary'))
            mhs_f1.append(f1_score(y_val, mhs.predict(scaled_X_val), average='binary'))

        print(f'{np.mean(lr_f1):.3f}')
        print(f'{np.mean(lrs_f1):.3f}')
        print(f'{np.mean(mhs_f1):.3f}')

    else:
        train, tweet_raw, test, s = read_data()
        tweet_processed = tweet_lower(train['text'])
        tweet_processed = tweet_stop_words(tweet_processed)
        #tweet_processed = tweet_stem(tweet_processed) # can't get this running
        train['text']=tweet_processed
        train.to_csv("cleaned_tweets.csv",index=False)

if __name__ == '__main__':
    freeze_support()
    main(20)
