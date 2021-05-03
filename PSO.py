import numpy as np


def Enxame(PAR,NPAR,MAX,MIN):
    x=np.zeros((NPAR, len(MAX)))
    for j in range(len(MAX)):
        for i in range(NPAR):
            x[i,j]=MIN[j]+(MAX[j]-MIN[j])*np.random.random()
    return x

def OBJ(x):
    rows = len(x)
    cols = len(x[0])
    fobj=np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            fobj[i]=FUNCTION(x[i,])
    return fobj
    # return fobj.min()

def VALOR(x):
    fob=FUNCTION(x)
    return fob

def BEST(X,RBEST):
    rows = len(X)
    cols = len(X[0])
    YCAL=OBJ(X)
    NEW=VALOR(RBEST)
    for i in range(rows):
        for j in range(cols):
            if(YCAL[i]<NEW):
                RBEST[j]=X[i,j]
    return RBEST

def PART(X,PBEST):
    rows = len(X)
    cols = len(X[0])
    YCAL=OBJ(PBEST)
    NEW=OBJ(X).min()
    for i in range(rows):
        for j in range(cols):
            if(YCAL[i]>NEW):
                PBEST[i,j]=X[i,j]
    return PBEST

def VE(X,VEL,BEST,PBEST,W,C1,C2):
    rows = len(X)
    cols = len(X[0])
    for i in range(rows):
        for j in range(cols):
            R1=np.random.random()
            R2=np.random.random()
            VEL[i,j]=W*VEL[i,j]+C1*R1*(PBEST[i,j]-X[i,j])+C2*R2*(BEST[j]-X[i,j])
            X[i,j]=X[i,j]+VEL[i,j]
    return VEL,X


def PSO(W,C1,C2,NPAR,ITE,PAR,MAX,MIN):
    X=Enxame(PAR,NPAR,MAX,MIN) # CRIA A POPULACAO
    ycal=OBJ(X) # CALCULA A FUNCAO OBJETIVO PARA TODAS PARTICULAS
    PBEST=PART(X,X) # O MELHOR LOCAL DE CADA PARTICULA INICIALMENTE ALEATORIA
    VBEST=[]
    for i in range(PAR):
        VBEST.append(1e10)
    VBEST=BEST(X,VBEST)
    VELOC=Enxame(PAR,NPAR,MAX,MIN)# VELOCIDADES INICIALMENTE ALEATORIAS
    RESP=[]
    for k in range(ITE):
        yteste=VALOR(VBEST)
        VELOC, X=VE(X,VELOC,VBEST,PBEST,W,C1,C2)
        VBEST=BEST(X,VBEST)
        PBEST=PART(X,PBEST)
        yteste=VALOR(VBEST)
        RESP.append(yteste)
    return RESP,VBEST
