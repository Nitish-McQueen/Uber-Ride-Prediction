#import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

#load CSV file
df=pd.read_csv("taxi.csv")
#Dependent and Independent Variables
x=df.iloc[:,:4].values
y=df.iloc[:,-1].values

# Model Creation
model=LinearRegression()
model.fit(x,y)

model.score(x,y)

model.predict([[30,1460000,9000,90]])

with open('uber.pkl','wb') as f:
    pickle.dump(model,f)



