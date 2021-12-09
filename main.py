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

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

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
        train_vecs = []
        for i in range(len(reals)):
            top_topics = (lda_model.get_document_topics(corpus[i],minimum_probability=0.0))
            topic_vec = [top_topics[i][1] for i in range(5)]
            #topic_vec.extend([reals.iloc[i].real_counts])
            topic_vec.extend([len(reals.text)])
            train_vecs.append(topic_vec)
        # Train Classifier
        X = np.array(train_vecs)
        y = np.array(reals.target)

        kf = KFold(5, shuffle=True, random_state=42)
        cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1,  = [], [], []

        for train_ind, val_ind in kf.split(X, y):
            # Assign CV IDX
            X_train, y_train = X[train_ind], y[train_ind]
            X_val, y_val = X[val_ind], y[val_ind]

            # Scale Data
            scaler = StandardScaler()
            X_train_scale = scaler.fit_transform(X_train)
            X_val_scale = scaler.transform(X_val)

            # Logisitic Regression
            lr = LogisticRegression(
                class_weight= 'balanced',
                solver='newton-cg',
                fit_intercept=True
            ).fit(X_train_scale, y_train)

            y_pred = lr.predict(X_val_scale)
            cv_lr_f1.append(f1_score(y_val, y_pred, average='binary'))

            # Logistic Regression SGD
            sgd = linear_model.SGDClassifier(
                max_iter=1000,
                tol=1e-3,
                loss='log',
                class_weight='balanced'
            ).fit(X_train_scale, y_train)

            y_pred = sgd.predict(X_val_scale)
            cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

            # SGD Modified Huber
            sgd_huber = linear_model.SGDClassifier(
                max_iter=1000,
                tol=1e-3,
                alpha=20,
                loss='modified_huber',
                class_weight='balanced'
            ).fit(X_train_scale, y_train)

            y_pred = sgd_huber.predict(X_val_scale)
            cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

        print(f'{np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
        print(f'{np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
        print(f'{np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')

    else:
        train, tweet_raw, test, s = read_data()
        tweet_processed = tweet_lower(train['text'])
        tweet_processed = tweet_stop_words(tweet_processed)
        #tweet_processed = tweet_stem(tweet_processed) # can't get this running
        train['text']=tweet_processed
        train.to_csv("cleaned_tweets.csv",index=False)

if __name__ == '__main__':
    freeze_support()
    main(50)
