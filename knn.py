from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score


frame = pd.read_csv("filedata_norm.csv", delimiter=',', index_col=0)

x_train = frame.iloc[:, :13].to_numpy()
y_train = frame.iloc[:, 13].to_numpy()

tuned_parametres = {

    "weights": np.array(["uniform", "distance"]),
    "algorithm": np.array(["ball_tree", "kd_tree", "brute"]),
    "p": np.array([2, 12]),
    }

model = GridSearchCV(KNeighborsClassifier(), tuned_parametres, cv=14)
# cv -- количество блоков кросс-валидации

model.fit(x_train, y_train)

best_model = model.best_estimator_

y_true = y_train
y_pred = best_model.predict(x_train)

precision = precision_score(y_true, y_pred, average='weighted')

print(precision)

if precision >= 0.95:

    with open("knn_params.pickle", "wb") as fout:

        pickle.dump(best_model.get_params(), fout)
        fout.close()

    with open("knn_best_model.sav", "wb") as fout:

        pickle.dump(best_model, fout)
        fout.close()