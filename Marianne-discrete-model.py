# -*- coding: utf-8 -*-

import numpy as np

############################# q1a     
def postD(nTotal, nWhite):
    def facto(n):
        fact=1
        for i in range(1,n+1):
            fact=fact*i
        return fact
    if (nWhite>nTotal or nWhite<0 or nTotal<0):
        return "Error in the parameters"
    else:
        n=10
        l=nTotal-nWhite
        post=np.zeros(n+1)
        like=np.zeros(n+1)
        prior=1./11
        ev=0
        for i in range(n+1):
            like[i]=(facto(nTotal)/(facto(l)*facto(nWhite)))*((i/n)**nWhite)*((1-(i/n))**l)
            ev+=like[i]*prior
        for i in range(n+1):
            post[i]=prior*like[i]/ev
        return post

############################# q1b      
def evidenceC(nTotal, nWhite):
    def facto(n):
        fact=1
        for i in range(1,n+1):
            fact=fact*i
        return fact
    if (nWhite>nTotal or nWhite<0 or nTotal<0):
        return "Error in the parameters"
    else:
        n=10
        l=nTotal-nWhite
        post=np.zeros(n+1)
        like=np.zeros(n+1)
        prior=np.array([1./6,0,1./6,0,1./6,0,1./6,0,1./6,0,1./6])
        ev=0
        for i in range(n+1):
            like[i]=(facto(nTotal)/(facto(l)*facto(nWhite)))*((i/n)**nWhite)*((1-(i/n))**l)
            ev+=like[i]*prior[i]
        return ev
