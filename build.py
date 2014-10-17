#from csv import DictReader
import csv
from datetime import datetime
#from csv import DictReader
from math import exp, log, sqrt
from sklearn.linear_model import SGDClassifier
import numpy as np
 
D = 2 ** 20   # number of weights use for learning
alpha = .1    # learning rate for sgd optimization

def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)

def get_x(csv_row, D):
    x = [0]  # 0 is the index of the bias term
    for count, value in enumerate(csv_row):
        index = int(value + str(count), 16) % D  # weakest hash ever ;)
        x.append(index)
    return x  # x contains indices of features that have a value of 1

def get_p(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)
        n[i] += 1.

    return w, n

w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature


clf = SGDClassifier(loss='modified_huber',)
#clf.partial_fit(np.zeros(D), np.asarray([0.,]), classes=np.asarray([0.,1.]))

loss = 0.
train = 'source/train_small.txt'
reader = csv.reader( open( train, 'r' ), delimiter = '\t' )
for t, row in enumerate(reader):
    if t > 100: 
        break

    y = 1. if row[0] == '1' else 0.

    del row[0]  # can't let the model peek the answer
    
    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, D)
    """
    # step 2, get prediction
    p = get_p(x, w)

    # for progress validation, useless for learning our model
    loss += logloss(p, y)
    if t % 10 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (
            datetime.now(), t, loss/t))

    # step 3, update model with answer
    w, n = update_w(w, n, x, p, y)
    """
    #SGDClassifier
    long_row = np.zeros(D)
    # add categorial values
    long_row[x] = 1.
    
    # fit model for one row
    clf.partial_fit(x, np.asarray([y]), classes=np.asarray([0.,1.]))

    # estimate classification
    pred = clf.predict(x)
    #print pred

    # estimate liklihood classification is 1
    pred_proba = clf.predict_proba(x)
    print pred_proba
    """
    loss += logloss(pred_proba[1], y)
    if t % 10 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), t, loss/t))
    """