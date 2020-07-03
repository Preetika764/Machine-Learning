import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

df = pd.read_csv("nasa.csv")

#print(df.head())

df.isnull().head()

#for col in df.columns:
#   print(len(df[col]))
    
#There is no missing data

#df.info()

df['Hazardous'].replace({True: 1, False: 0}, inplace = True)

df.drop(['Neo Reference ID', 'Name', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 
          'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Close Approach Date',  'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)', 'Orbiting Body', 'Orbit ID', 'Orbit Determination Date', 'Equinox'], axis=1, inplace = True)

#print(df.columns)

print("5")

y = df[['Hazardous']]

print(8)

df = normalize(df)

X = df[['Absolute Magnitude', 'Est Dia in KM(min)', 'Est Dia in KM(max)',
       'Epoch Date Close Approach', 'Relative Velocity km per sec',
       'Miss Dist.(kilometers)', 'Orbit Uncertainity',
       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
       'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
       'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly',
       'Mean Motion']]

#print(X[0:3])

#print(df['Hazardous'].value_counts())

#print(y[0:5])

#print(df.corr()['Hazardous'])

#df.info()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = SVC(C= 15,  kernel = 'rbf')

model = lm.fit(X_train, y_train)

y_predict= lm.predict(X_test)

print("Train score:")
print(lm.score(X_train, y_train))

print("Test score:")
print(lm.score(X_test, y_test))







