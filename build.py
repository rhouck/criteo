import csv

train = 'source/train.txt'

i_f = open( train, 'r' )
reader = csv.reader( i_f )

for index, line in enumerate(reader):
    if index > 1:
        break
    for l in line:
        print l 
 

