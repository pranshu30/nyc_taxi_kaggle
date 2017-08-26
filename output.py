import pandas as pd

col1 = pd.read_csv('sample_submission.csv')

col1 = col1.drop('click',axis=1)
#test = col1.iloc[:,0].values

col2 = pd.read_csv('outputlgb.txt',sep = " ")

sub = pd.DataFrame({'ID':col1['ID'], 'click':col2['click']})
sub.to_csv('o2.csv', index=False)