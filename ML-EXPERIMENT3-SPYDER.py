# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 13:37:21 2022

@author: home
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("record_irregular2.csv")
df

print(df)

plt.xlabel('VALUE OF X')
plt.ylabel('VALUE OF Y')
plt.scatter(df.xValue,df.yValue,color='red',marker='+')



new_df = df.drop('yValue',axis='columns')
new_df
print(new_df)

model = linear_model.LinearRegression()
model
print(model)

model.fit(new_df,df.yValue)
print(model.fit(new_df,df.yValue))

print(model.predict([[120]]))

print(model.predict([[150]]))

print(model.coef_)

print(model.intercept_)

print(3.05151515 * 120 -0.3333333333333144)

print(3.05151515 * 150 -0.3333333333333144)

xValueP_df = pd.read_csv("record_irregular2Predict.csv")
xValueP_df
print(xValueP_df)

p=model.predict(xValueP_df)
print(p)


xValueP_df['yValue']=p
xValueP_df
print(xValueP_df)

xValueP_df.to_csv("record_irregular2PredictGenerate4.csv")
xValueP_df
print(xValueP_df)

final_df = pd.read_csv("record_irregular2PredictGenerate4.csv")
final_df
print(final_df)

plot_df = pd.read_csv("record_irregular2PredictGenerate4.csv")
plot_df
print(plot_df)

plt.xlabel('VALUE OF X')
plt.ylabel('VALUE OF Y')
plt.scatter(plot_df.xValue,plot_df.yValue,color='red',marker='+')


