# -*- coding: utf-8 -*-

import numpy as np

############################# q3a-1     
def lml(alpha,beta,Phi,Y):
    pi=np.pi
    N=Phi.shape[0]
    B=alpha*Phi@ Phi.T+ beta*np.identity(N)
    logml=-(N/2)*np.log(2*pi)-(1/2)*np.log(np.abs(np.linalg.det(B)))-(1/2)*Y.T@ np.linalg.inv(B)@ Y
    return logml

############################# q3a-2      
def grad_lml(alpha,beta,Phi,Y):
    pi=np.pi
    N=Phi.shape[0]
    B=alpha*Phi@ Phi.T+ beta*np.identity(N)
    B1=np.linalg.inv(B)
    gradlml=np.zeros(2)
    gradlmla=-(1/2)*np.trace(Phi@ Phi.T@ B1)+(1/2)*(Y.T @ B1 @ Phi@ Phi.T@ B1@ Y)
    gradlmlb=-(1/2)*np.trace(B1)+(1/2)*(Y.T @ B1 @ B1@ Y)
    gradlml[0]=gradlmla
    gradlml[1]=gradlmlb
    return gradlml
