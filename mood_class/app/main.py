import pickle
from flask import Flask 

with open('model_pkl', 'rb') as f:
    data = pickle.load(f)

# flask setup
app = Flask(__name__)

@app.route("/getValue/<text>")
def getVal(text):
    return data.predict([text])[0]


