import pandas as pd
from preprocessing import *
import os
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint

import pyLDAvis.gensim_models
import pyLDAvis


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def main(num_topics):
    if(os.path.exists("cleaned_tweets.csv")):
        cleaned_tweets = pd.read_csv("cleaned_tweets.csv")
        reals=cleaned_tweets[cleaned_tweets['target']==1]
        print(reals['text'][:10])
        data = reals['text'].values.tolist()
        data_words = list(sent_to_words(data))
        id2word=corpora.Dictionary(data_words)
        texts=data_words
        corpus=[id2word.doc2bow(text) for text in texts]
        lda_model=gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
        pprint(lda_model.print_topics())
        doc_lda=lda_model[corpus]
        LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
    else:
        train, tweet_raw, test, s = read_data()
        tweet_processed = tweet_lower(train['text'])
        tweet_processed = tweet_stop_words(tweet_processed)
        #tweet_processed = tweet_stem(tweet_processed) # can't get this running
        train['text']=tweet_processed
        train.to_csv("cleaned_tweets.csv",index=False)


main(10)
