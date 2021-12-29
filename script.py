# Load Packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv("dataset/emotion_dataset_2.txt")

# remove stop words
#df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)
df['Clean_Text'] = df['Clean_Text'].apply(lambda x: np.str_(x))

# Features & Labels
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

#  Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)

# LogisticRegression Pipeline
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(solver='lbfgs', max_iter=400))])

# Train and Fit Data
pipe_lr.fit(x_train,y_train)

# Make A Prediction
# ex1 = "This book was so interesting it made me happy"

# print(pipe_lr.predict([ex1]))

# export model
with open('model_pkl', 'wb') as files:
    pickle.dump(pipe_lr, files)

