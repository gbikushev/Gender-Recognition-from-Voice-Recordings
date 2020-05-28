import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score

frame = pd.read_csv("filedata_norm.csv", delimiter=',', index_col=0)

x_train = frame.iloc[:, :13].to_numpy()
y_train = frame.iloc[:, 13].to_numpy()

tuned_parametres = {
    "n_estimators": np.array([50, 100, 150]),
    "criterion": np.array(["gini", "entropy"]),
    "max_features": np.array(["sqrt", "log2", None])
}

model = GridSearchCV(RandomForestClassifier(), tuned_parametres, cv=14)

model.fit(x_train, y_train)

best_model = model.best_estimator_

y_true = y_train
y_pred = best_model.predict(x_train)

precision = precision_score(y_true, y_pred, average='weighted')

print(precision)

if precision >= 0.95:

    with open("rfc_params.pickle", "wb") as fout:

        pickle.dump(best_model.get_params(), fout)
        fout.close()

    with open("rfc_best_model.sav", "wb") as fout:

        pickle.dump(best_model, fout)
        fout.close()
