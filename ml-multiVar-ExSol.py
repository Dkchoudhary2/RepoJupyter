# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 19:09:59 2022

@author: home
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

print('1.......................................................')
pd.read_csv('hiring.csv')
print(pd.read_csv('hiring.csv'))


print('2.......................................................')
d = pd.read_csv('hiring.csv')
d
print(d)

print('3.......................................................')
z=d.experience.fillna('zero')
print(z)

print('4.......................................................')
d.experience = d.experience.fillna('zero')
print(d)

print('5.......................................................')
z=d.experience.apply(w2n.word_to_num)
print(z)

print('6.......................................................')
d.experience = d.experience.apply(w2n.word_to_num)
print(d)


print('7.......................................................')
import math
median_test_score = math.floor(d['test_score(out of 10)'].mean())
print(median_test_score)


print('8.......................................................')
d['test_score(out of 10)'] = d['test_score(out of 10)'].fillna(median_test_score)
print(d)


print('9.......................................................')
reg = linear_model.LinearRegression()
reg.fit(d[['experience','test_score(out of 10)','interview_score(out of 10)']],d['salary($)'])
print(reg)


print('10.......................................................')
x=reg.predict([[2,9,6]])
print(x)


print('11.......................................................')
x=reg.predict([[12,10,10]])
print(x)






































