import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare(fake_df, true_df):
    fake_df["label"]=0
    true_df["label"]=1
    df=pd.concat([fake_df, true_df], ignore_index=True)
    return df
def cleanit():
    pass
def clean_text(text):
    stop_words=set(stopwords.words("English"))
    text=text.lower() #Converts the text to lower case
    text=re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) #removes the links i.e strings starting with http, www or https and replace it with empty strings '' or means delete the links
    text= text.translate(str.maketrans('','',string.punctuation))
    tokens=word_tokenize(text) # Tokenizes the text and returns a list of words from given text
    filtered_tokens=[word for word in tokens if word not in stop_words and word.isalpha()]  #Returns the list of tokens if the token is not present in list of stopwords and if token is alphabetic not number
    return " ".join(filtered_tokens)

def clean_data(df):
    df["clean_text"]=df["text"].apply(clean_text)
    verctorizer=TfidfVectorizer(max_features=5000)
    x=verctorizer.fit_transform(df["clean_text"])
    y=df["label"].values
    return x,y,verctorizer

