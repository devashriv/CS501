
from nltk.probability import FreqDist


def tweet_analysis():
    # Frequency distribution
    # incomplete
    fdist = FreqDist()

    # disaster names- nouns
    kwd_list1 = ["disaster", "earthquake", "volcano",
                 "apocalypse", "flood", "storm", "fire", "wildfire", "avalanche"]
    # disaster actions - verbs
    kwd_list2 = ["fall", "struck", "collapse", "bombard",
                 "shoot", "attack", "kill", "burst", "blast"]
    # disaster reactions by people - verbs
    kwd_list3 = ["help", "save", "pray"]

    # print(fdist["disaster"])
