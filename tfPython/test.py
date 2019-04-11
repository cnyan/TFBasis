#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2019/4/9 16:34
Describe：
    
    
"""
import numpy as np
import matplotlib.pyplot as plt

data_x1 = []
data_x2 = []
for i in range(200):
    x1 = np.random.uniform(-1, 1)
    data_x1.append(x1)
    x2 = np.random.uniform(0, 2)
    data_x2.append(x2)

plt.plot(data_x1)
plt.show()