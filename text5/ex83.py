# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:55:13 2020

@author: 15272
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
from math import *

data=pd.read_csv('D:/生存分析/text5/larynx.csv')#load data
Di=collections.Counter(data['time'])

def low(beta,Z):
    sum1=e**(np.dot(Z,beta))
    sum2=sum1*Z
    m,n=np.shape(Z)
    sum3=[[]]*m
    sum3[0]=sum1[0]**2*(Z[0:1,:].T @ Z[0:1,:])
    for i in range(1,m):
        sum3[i]=sum3[i-1]+sum1[i]**2*(Z[i:i+1,:].T @ Z[i:i+1,:])
    return sum1,sum2,sum3

def cum(j,Sn,z):
    m,n=np.shape(Sn)
    s=np.zeros((n,))
    for i in range(j):
        s+=Sn[i]#dim3+3
    if z=='add':
        return s
    else:
        s=np.matrix(s)
        s=np.array(s.T @ s)
        return s

def Newton(data,beta):
    n,m=np.shape(data)
    p=np.shape(beta)[0]
    L,grad,laplace=0,np.zeros((1,m-3)),np.zeros((p,p))
    Z,Ri,ti=df[:,m-p:m],df[:,1],df[:,0]
    sum1,sum2,sum3=low(beta,Z)
    for i in range(n):
        #L+=Ri[i]*(np.dot(beta,Z[i,:])-Di[ti[i]]*log(sum1))
        grad+=Ri[i]*(Z[i]-Di[ti[i]]*cum(i,sum2,'add')/np.cumsum(sum1)[i])
        laplace -= Ri[i]*Di[ti[i]]*(sum3[i]*np.cumsum(sum1)[i])-cum(i,sum2,'m')/(np.cumsum(sum1)[i])**2
    return np.dot(grad,np.linalg.inv(laplace)),grad

df=np.array(data)
n_iterations = 1000  #设置迭代次数
p=3
beta = np.random.randn(3,1)   #beta的数
for iteration in range(n_iterations):  #对迭代次数进行循环
    grads,Like = Newton(df,beta) #损失函数求偏导
    beta = beta - grads.T  #更新theta值
    if np.max(np.abs(Like))<0.01:
        print(iteration+1)
        break