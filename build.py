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

def prediction(p):
    pred = 1. if p >= 0.5 else 0.
    return pred 

loss = 0.
accuracy = 0.
clf = SGDClassifier(loss='log', alpha=0.001)
hasher = FeatureHasher(input_type='string', n_features=(2 ** 15))
chart_data = np.empty((0,3), float)

for t, row in enumerate(reader):
    
    if t > 0:
        break    

    y = 1. if row[0] == '1' else 0.
    del row[0]
    
    # step 1, get the hashed features
    x = hash_x(row, hasher)
    print x
    print type(x)
    if t > 1:
        
        # estimate liklihood classification is 1
        prob = clf.predict_proba(x)[0][1]
        # estimate classification
        pred = prediction(prob)
        # calc cost function and increment
        loss += logloss(prob, y)
        # add one point for correct prediction
        accuracy += 1. if pred == y else 0.
            

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

