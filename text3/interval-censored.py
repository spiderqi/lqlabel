# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:51:36 2020

@author: 15272
"""
from lifelines import NelsonAalenFitter, CoxPHFitter, KaplanMeierFitter
import pandas as pd
import re
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from IPython.display import display
from tabulate import tabulate

def alphaij(start,end):
    tao=list(set(start)|set(end))
    m,n=len(tao),len(end)
    alpha=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if tao[i]<=end[j] and tao[i]>start[j]:
                alpha[i][j]=1
            else:
                alpha[i][j]=0
    return alpha

def S(alpha,m,n):
    alphat=alpha.T
    a=[]
    for i in range(n):
        a.append(1/(alphat[i].sum()))
    p=np.dot(alpha,a)/46
    s=1-p.cumsum()
    for i in range(len(s)):
        s[i]=round(s[i],3)
    return s,p

def dj(alpha,p,m,n):  
    s=np.dot(alpha.T,p)#size n
    z=((alpha.T)*p).T/s    
    d=sum(z.T)
    for i in range(m):
        d[i]=round(d[i],3)
    Y=((d[::-1]).cumsum())[::-1]
    return d,Y

def add(St,p):
    for i in range(1,len(St)):
        p[i]=St[i]-St[i-1]
    return p        

def Sta(alpha,St,d,Y,p,e=0.001):
    m,n=alpha.shape
    Sa,k=[St],1
    St_hat=(1-d/Y).cumprod()
    while np.max(np.abs(St-St_hat))>e and k<1000:
        St=St_hat
        Sa.append(St)
        p=add(St,p)
        d,Y=dj(alpha,p,m,n)
        St_hat=(1-d/Y).cumprod()
        k+=1
    return Sa,k
    
def fun(x,mx=48):
    if x<=mx:
        return x
    else:
        return mx


#load data
df=pd.read_excel('D:/生存分析/text3/label3.xlsx')

#prepare
start=df[df['method']=='Ro']['starttime']
end=df[df['method']=='Ro']['endtime']
mx=max(end)
#数据处理
end=end.apply(lambda x:fun(x,mx))

alpha=alphaij(start,end)
m,n=alpha.shape
St,p=S(alpha,m,n)
alphat=alpha.T
d,Y=dj(alpha,p,m,n)
Sall,k=Sta(alpha,St,d,Y,p)
#table5.4
tao,St_hat=list(set(start)|set(end)),(1-d/Y).cumprod()
change=St-St_hat
tb=pd.DataFrame({'tao':tao,'St':St,'d':d,'Y':Y,'St_hat':St_hat,'change':change})

#data rac
start1=df[df['method']=='RaC']['starttime']
end1=df[df['method']=='RaC']['endtime']
mx1=max(end1)
end1=end1.apply(lambda x:fun(x,mx1))
start1,end1=np.array(start1),np.array(end1)

alpha1=alphaij(start1,end1)
m1,n1=alpha1.shape
St1,p1=S(alpha1,m1,n1)
alphat1=alpha1.T
d1,Y1=dj(alpha1,p1,m1,n1)
Sall1,k1=Sta(alpha1,St1,d1,Y1,p1)

tao1,St1_hat=list(set(start1)|set(end1)),(1-d1/Y1).cumprod()
change1=St1-St1_hat
tb=pd.DataFrame({'tao':tao1,'St':St1,'d':d1,'Y':Y1,'St_hat':St1_hat,'change':change1})

fig,ax=plt.subplots(figsize=(10,4))
#plot Survival function
ax.step(tao,St,where='post')
ax.step(tao1,St1,where='post')


#use package
kmf=KaplanMeierFitter()
t = np.linspace(0, 60, 61)

kmf.fit(end, event_observed=df[df['method'] == 'Ro']['have_death'], timeline=t, label="Ro")
kmf.plot()

kmf.fit(end1, event_observed=df[df['method'] == 'RaC']['have_death'], timeline=t, label="RaC")
kmf.plot()

plt.ylim(0,1)
plt.title("not survival rate between two treatment regimens")