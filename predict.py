import joblib
from scripts.preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
model=joblib.load("Models/randomforestmodel.pkl")
vectorizer=joblib.load("Models/tfidfvectorizer.pkl")


def predict(news_text):
    cleaned=clean_text(news_text)
    vectorized=vectorizer.transform([cleaned])
    prediction=model.predict(vectorized)
    label ="Real News 🟢" if prediction==1 else "Fake News 🔴"
    return label

if __name__=="__main__":
    print("\nFake News Detector")
    user_input=input("Paste the news content here")
    result=predict(user_input)
    print(result)
