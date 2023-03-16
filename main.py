import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import joblib
def logistic():
    a = pd.read_csv('parkfactor.csv', encoding='cp949')

    x = pd.DataFrame(a.iloc[:,[1,2,3,4]])
    y = pd.DataFrame(a.iloc[:,[5]])

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=0)
    LR = LogisticRegression()
    LR.fit(x,y)
    b = LR.predict(x_test)
    joblib.dump(LR, './KBO_park_homerun_predict.pkl')

if __name__ == "__main__":
    logistic()