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
X_train_res, y_train_res = smt.fit_resample(X, y)

X, y = shuffle(X_train_res, y_train_res)


model = SVC(C= 50, gamma= 'scale', kernel= 'rbf')
model.fit(X,y)


pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))