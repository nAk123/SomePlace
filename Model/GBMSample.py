import pandas as pd
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

data = pd.read_csv('SampleData.csv')
print data.shape
data.columns = ['id', 'srch_id','date_time','site_id','visitor_location_country_id','visitor_hist_starrating','visitor_hist_adr_usd','prop_country_id','prop_id',
'prop_starrating','prop_review_score','prop_brand_bool','prop_location_score1','prop_location_score2','prop_log_historical_price','position',
'price_usd','promotion_flag','srch_destination_id','srch_length_of_stay','srch_booking_window','srch_adults_count','srch_children_count',
'srch_room_count','srch_saturday_night_bool','srch_query_affinity_score','orig_destination_distance','random_bool','comp1_rate','comp1_inv',
'comp1_rate_percent_diff','comp2_rate','comp2_inv','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate',
'comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff',
'comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff','click_bool','gross_bookings_usd',
'booking_bool']
print 'Labels are'
d = data.to_string(max_rows=1)
print d 
print 'Split data into source and target sets'
x = data.drop(['id', 'date_time', 'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'], 1)
print 'Convert source to csv'
x.to_csv('SourceSample.csv')
#y = data[['position', 'click_bool', 'booking_bool']]
print 'Target data stored to csv'
x = x.fillna(0)
print 'Missing values in x have been filled'
y = data[['position']]
y.to_csv('TargetSample.csv')

print 'Split data into training and test sets'
offset = int(x.shape[0] * 0.9)
print offset
x_train, y_train = x[:offset], y[:offset]
x_test, y_test = x[offset:], y[offset:]
print len(x_test)
print len(y_test)


#Define GBM with following parameters
params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,
                  'learning_rate': 0.01, 'loss': 'ls'}
gbm = ensemble.GradientBoostingRegressor(**params)
#Train the GBM with the training set
print x_train.shape 
print y_train.shape
gbm.fit(x_train, y_train)
print 'GBM successfully run'
y_pred = pd.DataFrame(gbm.predict(x_test))

print 'Prediction made'
y_pred.to_csv('Predictions.csv')
y_test.to_csv('TestTarget.csv')
print y_test.as_matrix()
print y_pred.as_matrix()
print 'Created predictions table file'
mse = mean_squared_error(y_test, y_pred)
print("MAE: %.4f" % mse)
