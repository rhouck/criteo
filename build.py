#from csv import DictReader
import csv
from datetime import datetime
#from csv import DictReader
from math import exp, log, sqrt
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = 2 ** 10  # number of weights use for learning
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    #print p
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


train = 'source/train_small.txt'
reader = csv.reader( open( train, 'r' ), delimiter = '\t' )

"""
alpha = .1    # learning rate for sgd optimization
w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature

loss = 0.
accuracy = 0.
for t, row in enumerate(reader):
    #if t > 10: 
    #    break
    
    y = 1. if row[0] == '1' else 0.

    del row[0]  # can't let the model peek the answer
    
    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, D)
    
    # step 2, get prediction
    p = get_p(x, w)

    # estimate classification
    pred = 1. if p >= 0.5 else 0.
    if pred == y:
        accuracy += 1.

    # for progress validation, useless for learning our model
    loss += logloss(p, y)
    if t % 100 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f\tcurrent score: %f' % (datetime.now(), t, loss/t, accuracy/t))
        #print('%s\tencountered: %d\tcurrent logloss: %f' % (
        #    datetime.now(), t, loss/t))

    # step 3, update model with answer
    w, n = update_w(w, n, x, p, y)
"""









from sklearn.feature_extraction import FeatureHasher
import csv
from datetime import datetime
from math import exp, log, sqrt
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)

def hash_x(csv_row, hasher):
    """
    x = {}
    for count, value in enumerate(csv_row):
        x['%s' % count] = value
    """
    x = []
    for count, value in enumerate(csv_row):
        if value:
            x.append('F%s:%s' % (count,value))
    
    x = hasher.transform([x])
    return x 

loss = 0.
accuracy = 0.
clf = SGDClassifier(loss='log', alpha=0.001)
hasher = FeatureHasher(input_type='string', n_features=(2 ** 10))
chart_data = np.empty((0,3), float)
batch_size = 10
for t, row in enumerate(reader):
    

    y = 1. if row[0] == '1' else 0.

    del row[0]
    
    # main training procedure
    # step 1, get the hashed features
    
    x = hash_x(row, hasher)
    
    if t > 1:
        # estimate liklihood classification is 1
        p = clf.predict_proba(x)[0][1]
        loss += logloss(p, y)
        
        # estimate classification
        pred = 1. if p >= 0.5 else 0.
        if pred == y:
            accuracy += 1.

        if t % 1000 == 0:
            avg_loss = loss/t
            avg_accuracy = accuracy/t
            print('%s\tencountered: %d\tcurrent logloss: %f\tcurrent score: %f' % (datetime.now(), t, avg_loss, avg_accuracy))
            # track performance
            chart_data = np.append(chart_data, np.array([[t,avg_loss,avg_accuracy]]), axis=0)

        # add separate out integer features

        # scale integer featuers - rolling average / standard deviations
        #z = (x - mean) / std_dev


    # fit model for one row
    clf.partial_fit(x, np.asarray([y]), np.asarray([0.,1.]))

# print chart   
chart_data = pd.DataFrame({ 'Avg Loss': chart_data[:,1],
                            'Avg Accuracy': chart_data[:,2]}, 
                            index=chart_data[:,0])   

