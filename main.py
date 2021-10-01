import pandas as pd
from preprocessing import *


def main():

    train, tweet_raw, test, s = read_data()
    tweet_processed = tweet_split(tweet_raw)
    tweet_processed = tweet_lower(tweet_processed)
    tweet_processed = tweet_stop_words(tweet_processed)
    # tweet_processed = tweet_typo(tweet_processed) # this function takes too long time to run, may need improvement
    tweet_processed = tweet_stem(tweet_processed) # can't get this running


    # print(train)


main()
