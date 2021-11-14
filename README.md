# CS501

# **Introduction**:

This repository is our team's work towards CS501 project Fall 2021.
The aim of our project is to identify real disasters based on tweets, which may contain hyperbolic language, as accurately as possible. To achieve this we are using a training data set (train.csv) from Kaggle and developing our Natural Language Processing (NLP) model. We plan to validate our model by comparing it with the test dataset (test.csv) that has the correct results.


# **Documentation:**

A. Python scripts


0. main.py

Status: WIP

Description: Main script that will import functions from other scripts.


1. Exploratory Data Analysis.py: 

Status: complete

Description: This script performs preliminary analysis of the raw data. It grabs all the tweets from the training data set and generates visualizations. 


2. preprocessing.py

Status: complete

Description: This script contains functions that perform preprocessing on the raw data. It gets rid of stop words, symbols, converts everything to lower case. Essentially this is the first filter that cleans the raw data and converts it into a format that will be noise-free easy to manipulate in next steps.


3.  analysis.py

Status: The anaylsis code was included in main.py.

Description: This script contains the main NLP model that will guide the team in disaster identification.

Results: the output file is a dynamic interactive html file under results folder, named ldavis_prepared_10.html. It shows the Top-30 Most Salient Terms and an Intertopic Distance Map (via multidimensional scaling).



B. Data

1. test.csv : test data set that contains indexed tweets along with some location and keyword data (for some).
2. train.csv: train data set that contains indexed tweets along with some location and keyword data (for some) along with results of whether a disaster was identified or not as 0 or 1
3. sample_submission.csv

C. Report Generation
1. FIR_designp.pdf Output file
2. FIR_designp.pmd File that combines the report text and the code
3. FIR_designp.tex Intermediate file



# **Usage:**

1. Install dependencies

File/Math: pandas, numpy, matplot

Report generation: pweave, pandoc, LaTeX

NLP: nltk, autocorrect

To download specific NLTK resources-

    import nltk
    nltk.download()

select the items you want from the GUI and download

or similarly

    ntlk.download("autocorrect")
