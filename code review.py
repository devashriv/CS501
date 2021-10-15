##This is a test code to illustrate the function of the preprocessing part of our project.
##We used several text examples to illustrate each function.
##In the final code, we will read the raw text from an excel file. 

import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from autocorrect import Speller

def main():
    tweet_raw = ["Our Deeds are hte Reason of this #earthquake! May ALLAH Forgive us all.","we are cooking some dishes.","Forest fire near La Ronge Sask. Canada"]
    print(tweet_raw)
    tweet_processed = tweet_split(tweet_raw)
    print("split =",tweet_processed)
    tweet_processed = tweet_lower(tweet_processed)
    print("lower =",tweet_processed)
    tweet_processed = tweet_stop_words(tweet_processed)
    print("stop_words =",tweet_processed)
    tweet_processed = tweet_typo(tweet_processed) 
    print("typo =",tweet_processed)
    tweet_processed = tweet_stem(tweet_processed)
    print("stem =",tweet_processed)


def tweet_split(tweet_data):
    # 1. Split into words
    tweet_split = []
    for item in tweet_data:
        tweet_split += word_tokenize(item)

    return tweet_split

def tweet_lower(tweet_data):
    # 2. Text Pre-processing
    # 2.1 Make lower
    tweet_split_lower = []
    for word in tweet_data:
        tweet_split_lower.append(word.lower())

    return tweet_split_lower

def tweet_stop_words(tweet_data):
    # 2.2 Stop words - remove punctuations and random symbols
    stop_words = set(stopwords.words('english'))
    for word in tweet_data:
        if word in stop_words:
            tweet_data.remove(word)
            
    symbols = [".", ",", "!", "@", "#", "$", "%"]
    for word in tweet_data:
            if word in symbols:
                tweet_data.remove(word)

    return tweet_data

def tweet_typo(tweet_data):
    # 2.3 Typo correction - for example, correct "hte" to "the" 
    spell = Speller(lang='en')
    tweet_processed = []
    for word in tweet_data:
        correct_word = spell(word)
        tweet_processed.append(spell(word))

    return tweet_processed


def tweet_stem(tweet_data):
    # 2.4 Stemming - remove tense prefix/suffix and return root of word
    # use PorterStemmer to reduce a word to its word stem.
    # Now words such as “Likes”, ”liked”, ”likely” and ”liking” will be reduced to “like” after stemming.
    stemmer = PorterStemmer()
    tweet_stemmed = []
    for word in tweet_data:
        tweet_stemmed.append(stemmer.stem(word))
    return tweet_stemmed

main()
