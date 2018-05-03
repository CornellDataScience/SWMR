import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB

def get_web_y(X, Y, ranSeed, num_cv):
    web = []
    y = []
    random.seed(ranSeed)
    for i in range(num_cv):
        ind = (np.random.rand(len(Y)) <=(i+1)/5)&(np.random.rand(len(Y)) >i/5)
        web.append(X[ind])
        y.append(Y[ind])
    return web, y

def cv_man(web, y, clf):
    f1 = 0
    pre = 0
    recall = 0
    acc = 0
    for i in range(1,6):

        a = [0,1,2,3,4]
        a.remove(5-i)
        X_train = pd.concat(web[j] for j in a)
        y_train = pd.concat(y[j] for j in a)
        X_test = web[i-1]
        ytest = y[i-1]
        mnb = clf.fit(X_train, y_train)
        preds = mnb.predict(X_test)

        print ("f1_score for %d is %f" % (i, f1_score(ytest, preds, average='weighted')))
        print ("precision score for %d is %f" % (i, precision_score(ytest, preds, average='weighted')))
        print ("recall_score for %d is %f" % (i, recall_score(ytest, preds, average='macro')))
        print ("accuracy for %d is %f" % (i, accuracy_score(ytest, preds)))
        f1 = f1+f1_score(ytest, preds, average='weighted')
        pre = pre + precision_score(ytest, preds, average='weighted')
        recall = recall + recall_score(ytest, preds, average='macro')
        acc = acc + accuracy_score(ytest, preds)

    print ('average for acc', acc/5)
    print ('average for f1', f1/5)
    print ('average for pre', pre/5)
    print ('average for recall', recall/5)
