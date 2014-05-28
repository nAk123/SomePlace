import pandas as pd
import random

n = 5000000 #Number of observations in dataset
s = 100000 #Desired sample size
filename = "training_set_VU_DM_2014.csv"
print 'Random sample procedure'
skip = sorted(random.sample(xrange(n),n-s))
print 'Read the data into a data frame'
data = pd.read_csv(filename, skiprows=skip)
data.to_csv('SampleData.csv')

'''
print 'Split data into source and target sets'
x = data.drop(['date_time', 'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'], 1)
print 'Convert source to csv'
x.to_csv('SourceSample.csv')
#y = data[['position', 'click_bool', 'booking_bool']]
print 'Target data stored to csv'
y = data[['position']]
y.to_csv('TargetSample.csv')
print 'Done reading the file'
'''
