'''
split a file into two randomly, line by line. 
Usage: small_split.py <input file> <output file 1> <output file 2> [<limit>] [<probability of writing to the first file>] [<random seed>]'
Example: python small_split.py source/train.txt source/train_small.txt source/test_small.txt 1000 0.9
'''

import csv
import sys
import random

input_file = sys.argv[1]
output_file1 = sys.argv[2]
output_file2 = sys.argv[3]

try:
	limit = int( sys.argv[4] )
except IndexError:
	limit = 1000

try:
	P = float( sys.argv[5] )
except IndexError:
	P = 0.9
	
try:
	seed = sys.argv[6]
except IndexError:
	seed = None
	
print "P = %s" % ( P )

if seed:
	random.seed( seed )

i = open( input_file )
o1 = open( output_file1, 'wb' )
o2 = open( output_file2, 'wb' )

reader = csv.reader( i )
writer1 = csv.writer( o1 )
writer2 = csv.writer( o2 )

#headers = reader.next()
#writer1.writerow( headers )
#writer2.writerow( headers )

counter = 0

for line in reader:
	
	if counter >= limit:
		break

	r = random.random()
	if r > P:
		writer2.writerow( line )
	else:
		writer1.writerow( line )
	
	counter += 1
	if counter % 100000 == 0:
		print counter