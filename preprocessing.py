import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from autocorrect import Speller

def read_data():
    train = pd.read_csv('train.csv')
    s = pd.read_csv('sample_submission.csv')
    test = pd.read_csv('test.csv')
    tweet_raw = train['text']

    return train, tweet_raw, test, s

def tweet_split(tweet_data):
    # 1. Split into words
    tweet_split = []
    for item in tweet_data:
        tweet_split += word_tokenize(item)

    # print(tweet_split)
    return tweet_split

def tweet_lower(tweet_data):
    # 2. Text Pre-processing
    # 2.1 Make lower
    tweet_split_lower = []
    for word in tweet_data:
        tweet_split_lower.append(word.lower())


    # print(tweet_split_lower)
    return tweet_split_lower

def tweet_stop_words(tweet_data):
    # 2.2 Stop words - remove punctuations and random symbols
    #
    # Add function here
    #
    stop_words = set(stopwords.words('english'))
    for word in tweet_data:
        if word in stop_words:
            tweet_data.remove(word)

    print(tweet_data)
    return tweet_data

def tweet_typo(tweet_data):
    # 2.3 Typo correction - replace short forms with full
    # eg. "u" becomes "you", "wt" becomes "what"  Hang: # Rule based?
    #
    # print(spell('mussage')) ---> message
    # print(spell('survice')) ---> service
    # print(spell('hte')) ---> the
    #
    #
    spell = Speller(lang='en')
    tweet_processed = []
    for word in tweet_data:
        tweet_processed.append(spell(word))

    # print(tweet_processed)

    return tweet_processed


def tweet_stem(tweet_split):
    # 2.4 Stemming - remove tense prefix/suffix and return root of word
    word_lem = WordNetLemmatizer()
    i = 0
    while(i < len(tweet_split)):
        j = 0
        while(j < len(tweet_split[i])):
            tweet_split[i][j] = word_lem.lemmatize(tweet_split[i][j])
            print(tweet_split[i][j])
            j += 1
        i += 1
    print(tweet_split)
    return tweet_split
