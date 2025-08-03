from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier


def train(x,y):
    rf_model= RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    rf_model.fit(x, y)
    return rf_model