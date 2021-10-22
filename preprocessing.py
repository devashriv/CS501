import pandas as pd
import os
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from autocorrect import Speller


def read_data():
    train = pd.read_csv('train.csv')
    s = pd.read_csv('sample_submission.csv')
    test = pd.read_csv('test.csv')
    tweet_raw = train['text'][:10]

    return train, tweet_raw, test, s


def tweet_split(tweet_data):
    # 1. Split into words
    
    tweet_split = []
    for item in tweet_data:
        tweet_split += word_tokenize(item)
    # print(tweet_split)
    return tweet_split


def tweet_lower(tweet_data):
    #Convert all tweets to lowercase
    return ([x.lower() for x in tweet_data])


def tweet_stop_words(tweet_data):
    # 2.2 Stop words - remove stop words, URLs, punctuations and random symbols
    stops=stopwords.words('english')
    stops.extend(["im","theres"])
    for i in range(0,len(tweet_data)):
        st=tweet_data[i]
        st = re.sub(r'https?:\/\/.*\s*', '', st,)
        text_tokens = word_tokenize(st.translate(str.maketrans('', '', "!@#$%^&*.,/?;\"\"\'\'=><():")))
        tokens_without_sw = [word for word in text_tokens if not word in stops]
        tweet_data[i]=(" ").join(tokens_without_sw)
    return tweet_data


def tweet_typo(tweet_data):
    # 2.3 Typo correction - replace short forms with full
    # eg. "u" becomes "you", "wt" becomes "what"  Hang: # Rule based?
    #
    # print(spell('mussage')) ---> message
    # print(spell('survice')) ---> service
    # print(spell('hte')) ---> the
    #
    # this function takes too long time to run, may need improvement
    spell = Speller(lang='en')
    tweet_processed = []
    for word in tweet_data:
        correct_word = spell(word)
    #   print(correct_word)
        tweet_processed.append(spell(word))
    return tweet_processed


def tweet_stem(tweet_data):
    # 2.4 Stemming - remove tense prefix/suffix and return root of word
    # use PorterStemmer to reduce a word to its word stem.
    # Now words such as “Likes”, ”liked”, ”likely” and ”liking” will be reduced to “like” after stemming.
    stemmer = PorterStemmer()
    for i in range(0,len(tweet_data)):
        tweet_stemmed=[]
        st=word_tokenize(tweet_data[i])
        for word in st:
            tweet_stemmed.append(stemmer.stem(word))
        tweet_data[i]=(" ").join(tweet_stemmed)
    return tweet_data
