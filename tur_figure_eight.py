import sys, math
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
    pA=rA/(1+rA); pB=rB/(1+rB)
    T=np.zeros((V,V))
    for i in range(V):
        for j in range(V):
            if i==j: continue
            la,lb=dir_label(i,j)
            if la!=0: T[i,j]= pA if la>0 else 1-pA
            elif lb!=0: T[i,j]= pB if lb>0 else 1-pB
    return T/T.sum(axis=1,keepdims=True)
def stat(T):
    M=np.vstack([T.T-np.eye(V),np.ones(V)]); b=np.zeros(V+1); b[-1]=1
    p,*_=np.linalg.lstsq(M,b,rcond=None); p=np.clip(p,1e-15,None); return p/p.sum()
def kl(x,y):
    x=np.clip(x,1e-300,None); y=np.clip(y,1e-300,None)
    return float(np.sum(x*np.log(x/y)))
def quant(T):
    p=stat(T); w=p[:,None]*T
    DS=float(np.sum(np.where((w>1e-300)&(w.T>1e-300), w*np.log(np.clip(w,1e-300,None)/np.clip(w.T,1e-300,None)),0.0)))
    JS=0.0
    for i in range(V):
        a=T[i,:]; bb=(p*T[:,i])/p[i]; mm=(a+bb)/2
        JS+=0.5*p[i]*(kl(a,mm)+kl(bb,mm))
    return DS,JS
def tilt(T,z1,z2):
    M=np.zeros((V,V))
    for i in range(V):
        for j in range(V):
            if T[i,j]>1e-300:
                la,lb=dir_label(i,j); M[i,j]=T[i,j]*math.exp(z1*la+z2*lb)
    return M
def cd(T,h=1e-3):
    def mu(a,b): return max(np.real(np.linalg.eigvals(tilt(T,a,b))))
    J=np.array([(mu(h,0)-mu(-h,0))/(2*h),(mu(0,h)-mu(0,-h))/(2*h)])
    D=np.zeros((2,2))
    D[0,0]=0.5*(mu(h,0)-2*mu(0,0)+mu(-h,0))/(h*h)
    D[1,1]=0.5*(mu(0,h)-2*mu(0,0)+mu(0,-h))/(h*h)
    D[0,1]=D[1,0]=0.5*(mu(h,h)-mu(h,-h)-mu(-h,h)+mu(-h,-h))/(4*h*h)
    return J,D
print("="*102)
print("figure-eight（b1=2，两条独立电流）：本条 4JS vs 最优 TUR J^T D^-1 J")
print("="*102)
print("  %7s %7s %13s %13s %15s %10s %10s %8s" % ("rA","rB","sigma","4JS","J^T D^-1 J","本条/s","TUR/s","更紧"))
tot=win=0
for rA,rB in [(2,2),(2,1),(3,1),(5,1),(1,5),(10,1),(1,10),(10,10),(20,2),(2,20),(50,1),(1,50)]:
    T=build(rA,rB); DS,JS=quant(T); J,D=cd(T)
    mine=4*JS
    try: tur=float(J@np.linalg.solve(D,J))
    except Exception: tur=float('nan')
    tot+=1
    w = "本条" if mine>tur else "TUR"
    if mine>tur: win+=1
    print("  %7.2f %7.2f %13.6f %13.6f %15.6f %10.4f %10.4f %8s" % (rA,rB,DS,mine,tur,mine/DS,tur/DS,w))
print()
print("  总计 %d 例，本条胜 %d 例，TUR 胜 %d 例" % (tot,win,tot-win))
