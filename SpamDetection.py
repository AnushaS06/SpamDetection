from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('spam.csv',encoding='ISO-8859-1')

data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
data.head()
data.columns = ['labels', 'data']

data['b_labels'] = data['labels'].map({'ham': 0, 'spam': 1})
Y = data['b_labels'].values

obj = TfidfVectorizer()
X = obj.fit_transform(data['data'])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
def predict(name):
    model = name()
    model.fit(Xtrain, Ytrain)
    print("Classification rate for",name,":" , model.score(Xtest, Ytest))
    return model

def visualize(label):
    words = ''
    for msg in data[data['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


#can select the classification method

model = predict(MultinomialNB)
#model = predict(AdaBoostClassifier)
#model = predict(LogisticRegression)
visualize('spam')
visualize('ham')

yPred = model.predict(Xtest)
array = confusion_matrix(yPred,Ytest)
df_cm = pd.DataFrame(array)
plt.figure(figsize = (2,2))
sns.heatmap(df_cm, annot=True)

