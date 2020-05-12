# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:50:56 2020

@author: 15272
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:01:43 2020

@author: 15272
"""

import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import numpy as np

df=pd.read_excel('D:/生存分析/text4/table4.3.xlsx')#load data
dfl=pd.read_excel('D:/生存分析/text4/Aml.xlsx')
dfh=pd.read_excel('D:/生存分析/text4/Amh.xlsx')
def Nelson(di,yi):
    Ht=di/yi
    vht=di/(yi**2)
    return Ht.cumsum(),vht.cumsum()

def Epane(x,q,status=0):
    if status==1:
        q=1
    if x<=q and x>=-1:
        return (1-x**2)*3/4
    else:
        return 0

def kernel(t,time,h=100):
    s=(t-time)/h
    status=0
    if t<h:
        q=t/h
    elif t>662-h:
        q=(662-t)/h
        status=1
    else:
        q=1
    ks=s.apply(lambda x:Epane(x,q,status))
    alpha=64*(2-4*q+6*q**2-3*q**3)/((1+q)**4*(19-18*q+3*q**2))
    beta=240*(1-q)**2/((1+q)**4*(19-18*q+3*q**2))
    if status==1:
        s1=alpha+beta*(-s)
    else:
        s1=alpha+beta*s
    return s,ks*s1

def hhat(time,dHt,dVht,h=100):
    n=len(time)
    t=np.linspace(0,time[n-2],time[n-2])
    h1,vh1=[0]*len(t),[0]*len(t)
    for i in range(len(t)):
        s,ks=kernel(t[i],time)
        h1[i]=np.sum(ks*dHt)/h
        vh1[i]=np.sum(ks**2*dVht)/(h**2)
    return t,h1,vh1


Ht,Vht=Nelson(df['di'],df['Yi'])
Ht_high,Vht_high=Nelson(dfh['di'],dfh['Yi'])
Ht_low,Vht_low=Nelson(dfl['di'],dfh['Yi'])

sigmaH=np.sqrt(Vht)#standard 
dHt,dVht=Ht.diff(),Vht.diff()#calculate differ sequence
dHt[0],dVht[0]=Ht[0],Vht[0]
t1,kt1=kernel(150,df['ti'])
t2,kt2=kernel(50,df['ti'])
t3,kt3=kernel(600,df['ti'])
dt=pd.DataFrame({'time':df['ti'],'dHt':dHt,'dVht':dVht,'150-ti/100':t1,'K(150-ti/100)':kt1,
               '50-ti/100':t2,'K(50-ti/100)':kt2,'600-ti/100':t3,'K(600-ti/100)':kt3,})

dht_high,dvht_high,=Ht_high.diff(),Vht_high.diff()
dht_low,dvht_low,=Ht_low.diff(),Vht_low.diff()
#Figure 6.1
t,h1,vh1=hhat(df['ti'],dHt,dVht)
th,hh,vhh=hhat(dfh['ti'],dht_high,dvht_high)
tl,hl,vhl=hhat(dfl['ti'],dht_low,dvht_low)
plt.plot(t,h1)
plt.plot(th[0:len(t)],hh[0:len(t)])
plt.plot(tl[0:len(t)],hl[0:len(t)])