# CS501

**Introduction**:

This repository is our team's work towards CS501 project Fall 2021.
The aim of our project is to identify occurance of a disaster based on tweets as precisely as possible. To achieve this we are using a training data set (train.csv) from Kaggle and developing our Natural Language Processing (NLP) model. Finally we plan to validate our model by comparing it with the test dataset (test.csv) that has the correct results.


**Documentation:**

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

Status: not started

Description: This script contains the main NLP model that will guide the team in disaster identification. 


B. Data

1. test.csv : test data set that contains indexed tweets along with some location and keyword data (for some).
2. train.csv: train data set that contains indexed tweets along with some location and keyword data (for some) along with results of whether a disaster was identified or not as 0 or 1
3. sample_submission.csv

C. Report Generation
1. FIR_designp.pdf
2. FIR_designp.pmd
3. FIR_designp.txt



**Usage:**

1. Install dependencies

File/Math: pandas, numpy, matplot

Report generation: pweave, pandoc, LaTeX

NLP: nltk, autocorrect

To download specific NLTK resources-
>>>import nltk

>>>nltk.download()

select the items you want from the GUI and download
