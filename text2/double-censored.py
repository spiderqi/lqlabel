# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:27:57 2020

@author: 15272
"""

from lifelines import NelsonAalenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import re
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from IPython.display import display
from tabulate import tabulate


        
def stepk1(d,rc):
    sumd,sumrc=d[::-1].cumsum(),rc[::-1].cumsum()#计算累加和
    Y=(sumd+sumrc)[::-1]
    St=round((d/Y).apply(lambda x:1-x).cumprod(),3)
    return Y,St

def stepk2(St,p):
    t=list(St)
    t.insert(0,1)
    for i in range(1,n):
        for j in range(i):
            p[i-1][j]=round((t[j]-t[j+1])/(1-t[i]),3)
    return p

def stepk3(lc,d,p):
    x=[]
    for i in range(len(d)):
        sumc=0
        for j in range(i,len(d)):
            sumc+=p[j][i]*lc[j]         
        x.append(sumc)
    return d+x

def algorithm(a,e=0.001):
    try:
        k,Skt,Yk=1,[a],[Y]
        Y0,St0=Y,a
        Y,a=stepk1(d,rc)
        while max((a-St0).abs())>e and k<1000:
            Skt.append(a)
            Yk.append(Y)
            k+=1
            p=stepk2(a,p)
            d=stepk3(lc,d,p)
            Y0,St0=Y,a
            Y,a=stepk1(d,rc)
    except:
        print('wrong')
#load data
df=pd.read_excel('D:/生存分析/text2/label2.xlsx')
df.head()
lc,rc,d=df['Left Censored'],df['Right Censored'],df['NoEO']
Y,St=stepk1(d,rc)
df['Y']=Y
df['St']=St
df=df.append([{'i':'total','Age':' ','Left Censored':12,'NoEO':100,'Right Censores':79,'Y':0,'St':' '}],ignore_index=True)

#参数设定
n=len(St)+1
col=list('123456789')
p=pd.DataFrame(np.identity(n-1))
col.append('10')
p.columns=col

#al
p=stepk2(St,p)
d=stepk3(lc,d,p)
Skt,Yk=algorithm(St)

