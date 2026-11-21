import sys, math, itertools
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
V=5
def dir_label(i,j):
    la=lb=0
    if (i,j) in [(0,1),(1,2),(2,0)]: la=1
    if (i,j) in [(1,0),(2,1),(0,2)]: la=-1
    if (i,j) in [(0,3),(3,4),(4,0)]: lb=1
    if (i,j) in [(3,0),(4,3),(0,4)]: lb=-1
    return la,lb
def build(rA,rB):
    pA=rA/(1+rA); pB=rB/(1+rB); T=np.zeros((V,V))
    for i in range(V):
        for j in range(V):
            if i==j: continue
            la,lb=dir_label(i,j)
            if la!=0: T[i,j]=pA if la>0 else 1-pA
            elif lb!=0: T[i,j]=pB if lb>0 else 1-pB
    return T/T.sum(axis=1,keepdims=True)
def stat(T):
    M=np.vstack([T.T-np.eye(V),np.ones(V)]); b=np.zeros(V+1); b[-1]=1
    p,*_=np.linalg.lstsq(M,b,rcond=None); p=np.clip(p,1e-15,None); return p/p.sum()
def kl(x,y):
    x=np.clip(x,1e-300,None); y=np.clip(y,1e-300,None); return float(np.sum(x*np.log(x/y)))
def quant(T):
    p=stat(T); w=p[:,None]*T
    DS=float(np.sum(np.where((w>1e-300)&(w.T>1e-300), w*np.log(np.clip(w,1e-300,None)/np.clip(w.T,1e-300,None)),0.0)))
    JS=0.0
    for i in range(V):
        a=T[i,:]; bb=(p*T[:,i])/p[i]; mm=(a+bb)/2
        JS+=0.5*p[i]*(kl(a,mm)+kl(bb,mm))
    return DS,JS
def cd(T,h=1e-4):
    def mu(z1,z2):
        M=np.zeros((V,V))
        for i in range(V):
            for j in range(V):
                if T[i,j]>1e-300:
                    la,lb=dir_label(i,j); M[i,j]=T[i,j]*math.exp(z1*la+z2*lb)
        return max(np.real(np.linalg.eigvals(M)))
    J=np.array([(mu(h,0)-mu(-h,0))/(2*h),(mu(0,h)-mu(0,-h))/(2*h)])
    D=np.zeros((2,2))
    D[0,0]=0.5*(mu(h,0)-2*mu(0,0)+mu(-h,0))/(h*h)
    D[1,1]=0.5*(mu(0,h)-2*mu(0,0)+mu(0,-h))/(h*h)
    D[0,1]=D[1,0]=0.5*(mu(h,h)-mu(h,-h)-mu(-h,h)+mu(-h,-h))/(4*h*h)
    return J,D
print("="*84)
print("M2 所需：$(r_A,r_B)=(50,1)$ 的精确 sigma, J, D_J")
print("="*84)
for (rA,rB) in [(50,1),(10,10),(5,1)]:
    T=build(rA,rB); DS,JS=quant(T); J,D=cd(T)
    tur=float(J@np.linalg.solve(D,J))
    print("  (rA,rB)=(%g,%g)" % (rA,rB))
    print("    sigma = %.10f   (精确)" % DS)
    print("    JS    = %.10f   4*JS = %.10f   (4JS/sigma = %.6f)" % (JS,4*JS,4*JS/DS))
    print("    J     = [%.10e, %.10e]" % (J[0],J[1]))
    print("    D_J   = [[%.6e, %.6e],[%.6e, %.6e]]  (D=mu''/2, 使 Var(J_t)~2 D t)" % (D[0,0],D[0,1],D[1,0],D[1,1]))
    print("    J^T D^-1 J = %.10f   (TUR/sigma = %.6f)" % (tur,tur/DS))
    print("    比值 4JS/TUR = %.6f" % (4*JS/tur))
    print()
