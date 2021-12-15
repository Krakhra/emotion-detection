# Load Packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import neattext.functions as nfx
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

print(get_sentiment("I love coding"))