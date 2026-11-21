import sys, math, itertools
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
m=3; K=2; vals=[-1.0,1.0]; dp=2*math.pi/m
r=-1.0; u=1.0; D=1.0; beta=1.0
states=list(itertools.product(range(K),repeat=m)); idx={s:i for i,s in enumerate(states)}; N=len(states)
def H(s):
    A=[vals[k] for k in s]
    e=sum((r/2*A[i]**2+u/4*A[i]**4)*dp for i in range(m))
    e+=0.5*D*sum(((A[(i+1)%m]-A[i])/dp)**2*dp for i in range(m))
    return e
E=np.array([H(s) for s in states])
def metro():
    k=np.zeros((N,N))
    for si,s in enumerate(states):
        for i in range(m):
            t=list(s); t[i]=1-t[i]; t=tuple(t); tj=idx[t]
            k[si,tj]=min(1.0,math.exp(-beta*(E[tj]-E[si])))
    for i in range(N):
        sm=k[i].sum(); k[i]=k[i]/sm if sm>0 else np.eye(N)[i]
    return k
def add_drive(k,f,eps=0.3):
    kd=k.copy(); singles=[idx[(1,0,0)],idx[(0,1,0)],idx[(0,0,1)]]
    for j in range(3):
        a=singles[j]; b=singles[(j+1)%3]
        kd[a,b]+=eps*math.exp(f/2); kd[b,a]+=eps*math.exp(-f/2)
    for i in range(N): kd[i]/=kd[i].sum()
    return kd
def stat(T):
    M=np.vstack([T.T-np.eye(N),np.ones(N)]); b=np.zeros(N+1); b[-1]=1
    p,*_=np.linalg.lstsq(M,b,rcond=None); p=np.clip(p,1e-15,None); return p/p.sum()
def kl(x,y):
    x=np.clip(x,1e-300,None); y=np.clip(y,1e-300,None); return float(np.sum(x*np.log(x/y)))
def funcs(T):
    p=stat(T); w=p[:,None]*T
    DS=float(np.sum(np.where((w>1e-300)&(w.T>1e-300), w*np.log(np.clip(w,1e-300,None)/np.clip(w.T,1e-300,None)),0.0)))
    # 正确：H(pi)=sum_{i<j} (sqrt w_ij - sqrt w_ji)^2
    Hpi=0.0
    for i in range(N):
        for j in range(i+1,N):
            Hpi+=(math.sqrt(max(w[i,j],0))-math.sqrt(max(w[j,i],0)))**2
    JS=0.0
    for i in range(N):
        a=T[i,:]; bb=(p*T[:,i])/max(p[i],1e-300); mm=(a+bb)/2
        JS+=0.5*p[i]*(kl(a,mm)+kl(bb,mm))
    I=sum(p[i]*kl(T[i,:],p) for i in range(N) if p[i]>1e-300)
    A=1-math.sqrt(max(JS,0)/math.log(2))
    return DS,Hpi,JS,I,A
print("="*94)
print("修正版：环上 GL 场（m=3,K=2,8 态）；H(pi) 按 i<j 求和")
print("="*94)
print("  %6s %13s %13s %13s %13s %9s %9s %10s" % ("f","DeltaS","4H(pi)","4JS","I","A","JS/DS","检验"))
base=metro()
for f in (0.0,0.25,0.5,1.0,2.0,4.0):
    T=add_drive(base,f); DS,Hpi,JS,I,A=funcs(T)
    ok = (DS>=4*Hpi-1e-12) and (Hpi>=JS-1e-12) and (DS>=4*JS-1e-12)
    tag = "OK" if ok else "VIOLATION"
    r1 = (JS/DS) if DS>1e-14 else float('nan')
    print("  %6.2f %13.6e %13.6e %13.6e %13.6e %9.4f %9.4f %10s" % (f,DS,4*Hpi,4*JS,I,A,r1,tag))
print()
print("  判据：A 随 f 单调降；近平衡 JS/DS -> 0.25；定理三层不等式均 OK。")
