import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Global variables - raw data
train = pd.read_csv('train.csv')
s = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test.csv')
tweet_raw = train['text'][:10]
tweet_split = []


def tweet_split(tweet_raw):
    # 1. Split into words
    for item in tweet_raw:
        tweet_split.append(word_tokenize(item))
        tweet_split += word_tokenize(item)

    print(tweet_split)
    return tweet_split

def tweet_lower(tweet_split):
    # 2. Text Pre-processing
    # 2.1 Make lower
    i = 0
    while(i < len(tweet_split)):
        j = 0
        while(j < len(tweet_split[i])):
            tweet_split[i][j] = tweet_split[i][j].lower()
            j += 1
        i += 1

    print(tweet_split)
    return tweet_split

def tweet_stop_words():
    # 2.2 Stop words - remove punctuations and random symbols
    #
    # Add function here
    #
    stop_words = set(stopwords.words('english'))

    print(tweet_split)


def tweet_typo():
    # 2.3 Typo correction - replace short forms with full
    # eg. "u" becomes "you", "wt" becomes "what"
    #
    # Add function here
    #
    print(tweet_split)


def tweet_stem():
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
