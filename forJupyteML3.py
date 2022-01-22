# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 20:32:54 2022

@author: home
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 19:40:24 2022

@author: home
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("homeprices2.csv")
df

print(df)

print('1................OK............................................')
#%matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')



"""
print('2.......................NO NEED OF print()stm.....................................')
%matplotlib inline
print(plt.xlabel('area'))
print(plt.ylabel('price'))
print(plt.scatter(df.area,df.price,color='red',marker='+'))

"""

print('3........................')
new_df = df.drop('price',axis='columns')
print(new_df)


print('4.....................................')
model = linear_model.LinearRegression()
print(model)

print('5.............................')
model.fit(new_df,df.price)
print(model.fit(new_df,df.price))

print('6......................................')
print(model.predict([[3300]]))

print(model.predict([[1100]]))

print(model.predict([[5500]]))

print(model.predict([[8800]]))


print('7........................')
print(model.coef_)

print(model.intercept_)

print('8............HOW ML WORKS INTERNALLY................')

print(135.78767123*3300 + 180616.43835616432)
print(135.78767123*1100 + 180616.43835616432)
print(135.78767123*5500 + 180616.43835616432)
print(135.78767123*8800 + 180616.43835616432)

print('9...................................')
area_df = pd.read_csv("areas2.csv")
print(area_df)

print('10..............................')
p=model.predict(area_df)
print(p)

print('11...............................')
area_df['prices']=p
print(area_df)

print('12...........................')
area_df.to_csv("prediction2.csv")

