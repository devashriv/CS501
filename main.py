import pandas as pd


def main():

    train = pd.read_csv('train.csv')
    s = pd.read_csv('sample_submission.csv')
    test = pd.read_csv('test.csv')

    print(train)


main()
