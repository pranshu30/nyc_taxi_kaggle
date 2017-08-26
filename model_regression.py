import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop("dropoff_datetime",axis = 1)
import numpy as np
import time
import datetime


#Convert test time to timestamp(unix code)
def convert_to_timestamp2(date):
    date_as_string = str(date).strip()
    return time.mktime(datetime.datetime.strptime(date_as_string,"%Y-%m-%d %H:%M:%S").timetuple())

   
train['pickup_datetime']= train['pickup_datetime'].apply(convert_to_timestamp2)
test['pickup_datetime']= test['pickup_datetime'].apply(convert_to_timestamp2)

#Train
Y_train = train.iloc[:,9].values
X_train = train.iloc[:,1:9].values
#Test
X_test = test.iloc[:,1:9].values

from sklearn.preprocessing import LabelEncoder
label1 = LabelEncoder()
label2 = LabelEncoder()
X_train[:,7] = label1.fit_transform(X_train[:,7])
X_test[:,7] = label1.fit_transform(X_test[:,7])


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X_train[:, 0:8])
X_train[:, 0:8] = imputer.transform(X_train[:, 0:8])



from sklearn.preprocessing import Imputer
imputert = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputert = imputert.fit(X_test[:, 0:8])
X_test[:, 0:8] = imputert.transform(X_test[:, 0:8])



#


from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train, Y_train)

#--------------XGBOOST MODEL------------------
"""from xgboost import XGBClassifier
 classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

"""

#--------------------------LGBM MODEL-----------------------

"""from lightgbm import LGBMClassifier
classifier = LGBMClassifier()
classifier.fit(X_train, Y_train)
"""
y_pred = classifier.predict(X_test)


df = pd.DataFrame(y_pred)


df.to_csv('outpuut.csv',sep=' ',index=False, header=False)

