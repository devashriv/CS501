import pandas as pd
from preprocessing import *
import os
import re


def main():

    if(os.path.exists("cleaned_tweets.csv")):
        cleaned_tweets = pd.read_csv("cleaned_tweets.csv")
        split_corpus = tweet_split(cleaned_tweets['text'])
    else:
        train, tweet_raw, test, s = read_data()
        tweet_processed = tweet_lower(train['text'])
        tweet_processed = tweet_stop_words(tweet_processed)
        print(tweet_processed[1:10])
        tweet_processed = tweet_stem(tweet_processed) # can't get this running
        print(tweet_processed[1:10])
        train['text']=tweet_processed
        train.to_csv("cleaned_tweets.csv",index=False)


main()
