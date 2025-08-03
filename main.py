import joblib
import os
from scripts.data_loader import load_data
from scripts.preprocess import prepare, clean_data
from scripts.train_model import train
import time

def main():
    start_time = time.time()
    fake_df=load_data("./data/fake.csv")
    true_df=load_data("./data/True.csv")

    df=prepare(fake_df, true_df)
    x_clean,y_clean,vectorizer=clean_data(df)
    model =train(x_clean,y_clean)

    os.makedirs("Models",exist_ok=True)
    joblib.dump(model,"Models/randomforestmodel.pkl")
    joblib.dump(vectorizer, "Models/tfidfvectorizer.pkl")
    print(f"🏁 Pipeline finished in {time.time() - start_time:.2f} seconds.\n")
if __name__=="__main__":
    main()