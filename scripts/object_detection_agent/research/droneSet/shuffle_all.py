import glob
import csv
from random import shuffle
with open('droneset_test.csv','r') as ip:
	data=ip.readlines()
header, rest=data[0], data[1:]
shuffle(rest)
with open('droneset_test_shuffled.csv','w') as out:
	out.write(''.join([header]+rest))
