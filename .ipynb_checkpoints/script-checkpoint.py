# Load Packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import neattext.functions as nfx
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv("dataset/twitter.txt")

