import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as sk
import copy
import warnings
import pickle
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score,make_scorer
from numpy.random import seed
seed(1)

df = pd.read_csv('covid-dataset.csv')

df = df.astype({"label":'string'})
X=df.drop("label",axis=1).values[0:399]
X=X.astype(int)
y=df["label"].values[0:399]
print(X.shape)
print(X,y)

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)

from imblearn.over_sampling import SMOTE

smt = SMOTE(k_neighbors = 2)
X_train_res, y_train_res = smt.fit_sample(X, y)

X, y = shuffle(X_train_res, y_train_res)

for i in [10]:
  originalclass = []
  predictedclass = []
  def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)
  model = KNeighborsClassifier()
  n_neighbors = range(1, 21, 2)
  weights = ['uniform', 'distance']
  metric = ['euclidean', 'manhattan', 'minkowski']
  grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
  cv = StratifiedKFold(n_splits=i, shuffle=True, random_state=1)
  grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
  grid_search.fit(X,y)
  print(grid_search.best_params_)
  nested_score = cross_val_score(grid_search, X, y, cv=cv, scoring=make_scorer(classification_report_with_accuracy_score)) 
  print(classification_report(originalclass, predictedclass)) 
  print(np.mean(nested_score))


pickle.dump(grid_search,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))