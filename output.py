import pandas as pd

col1 = pd.read_csv('sample_submission.csv')

col1 = col1.drop('trip_duration',axis=1)
#test = col1.iloc[:,0].values

col2 = pd.read_csv('output.csv',header = None)

sub = pd.DataFrame({'ID':col1['id'], 'trip_duration':col2['trip_duration']})
sub.to_csv('submission1.csv', index=False)