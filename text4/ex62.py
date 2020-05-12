# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:41:18 2020

@author: 15272
"""

import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import numpy as np

df=pd.read_excel('D:/生存分析/text4/kidney_transplant.xlsx')#load data

use=collections.Counter(df[df['death']>0]['time'])
count=collections.Counter(df['time'])

def event(time,use,count):
    a=[]
    x=list(set(time))
    x.sort()
    n=len(x)
    d=[0]*n
    d[0]=len(time)
    for i in range(len(x)-1):
        d[i+1]=d[i]-count[x[i]]
        if x[i] in use.keys():
            a.append([x[i]/365,count[x[i]],d[i+1]])
    if x[-1] not in use.keys():
        a.append([x[-1]/365,0,d[-1]])
    a=pd.DataFrame(a,columns=['ti','di','Yi'])
    return a

def Nelson(di,yi):
    Ht=di/yi
    vht=di/(yi**2)
    return Ht.cumsum(),vht.cumsum()

def Kerf(x,q,f,status=0):
    if status==1:
        s=-1
    if x<=1 and x>=-1:
        if f=='uniform':
            return 1/2
        elif f=='Epane':
            return (1-x**2)*3/4
        elif f=='biweight':
            return ((1-x**2)**2)*15/16
        else:
            print(error)
    else:
        return 0

def coef(q, f):
    if f=='uniform':
        alpha=8*(1+q**3)/((1+q)**4)
        beta=12*(1-q)/(1+q**3)
    elif f=='Epane':
        alpha=64*(2-4*q+6*q**2-3*q**3)/((1+q)**4*(19-18*q+3*q**2))
        beta=240*(1-q)**2/((1+q)**4*(19-18*q+3*q**2))
    elif f=='biweight':
        alpha=64*(8-24*q+48*q**2-45*q**3+15*q**4)/((1+q)**5*(81-168*q+126**q-40*q**3+5*q**4))
        beta=1120*(1-q)**3/((1+q)**5*(81-168*q+126**q-40*q**3+5*q**4))
    else:
        alpha,beta=0,0
    return alpha,beta

def kernel(t,time,tD,f,h):
    s=(t-time)/h
    status=0
    if t<h:
        q=t/h
    elif t>tD-h:
        q=(tD-t)/h
        status=1
    else:
        q=1
    ks=s.apply(lambda x:Kerf(x,q,f,status))
    alpha,beta=coef(q,f)
    if status==1:
        s1=alpha+beta*(-s)
    else:
        s1=alpha+beta*s
    return s,ks*s1

def hhat(time,dHt,dVht,tD,K,h):
    n=len(time)
    t=np.linspace(0,time[n-2],700)
    H,V=[],[]
    for f in K:
        h1,vh1=[0]*len(t),[0]*len(t)
        for i in range(len(t)):
            s,ks=kernel(t[i],time,tD,f,h)
            h1[i]=np.sum(ks*dHt)/h
            vh1[i]=np.sum(ks**2*dVht)/(h**2)
        H.append(h1)
        V.append(vh1)
    return t,H,V

#figure 1
Df=event(df['time'],use,count)
Ht,Vht=Nelson(Df['di'],Df['Yi'])
plt.step(Df['ti'],Ht)

#figure 2
dHt,dVht=Ht.diff(),Vht.diff()
dHt[0],dVht[0]=Ht[0],Vht[0]
tD=max(Df[Df['di']>0]['ti'])

K=['uniform','Epane','biweight']
t,H,V=hhat(Df['ti'],dHt,dVht,tD,K,1)
plt.xlim(0,8.2)
plt.plot(t,H[0])
plt.plot(t,H[1])
plt.plot(t,H[2])

#figure 3

t1,H1,V1=hhat(Df['ti'],dHt,dVht,tD,K,0.5)
t2,H2,V2=hhat(Df['ti'],dHt,dVht,tD,K,2)
t3,H3,V3=hhat(Df['ti'],dHt,dVht,tD,K,1.5)
plt.xlim(0,8.2)
plt.plot(t1,H1[1])
plt.plot(t,H[1],'--')
plt.plot(t2,H2[1],'-.')
plt.plot(t3,H3[1],'--')

