from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv('spambase/spambase.data').values
np.random.shuffle(data) 

X = data[:,:48]
Y = data[:,-1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for NB:", model.score(Xtest, Ytest))

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate for LogisticRegression:", model.score(Xtest, Ytest))

yPred = model.predict(Xtest)
array = confusion_matrix(yPred,Ytest)
df_cm = pd.DataFrame(array)
plt.figure(figsize = (2,2))
sns.heatmap(df_cm, annot=True)
