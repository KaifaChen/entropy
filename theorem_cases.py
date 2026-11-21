import sys, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
def stat(T):
    m=T.shape[0]
    M=np.vstack([T.T-np.eye(m),np.ones(m)]); b=np.zeros(m+1); b[-1]=1
    p,*_=np.linalg.lstsq(M,b,rcond=None); p=np.clip(p,1e-15,None); return p/p.sum()
def kl(x,y):
    x=np.clip(x,1e-300,None); y=np.clip(y,1e-300,None)
    return float(np.sum(x*np.log(x/y)))
def quant(T):
    m=T.shape[0]; p=stat(T); w=p[:,None]*T
    DS=float(np.sum(np.where((w>1e-300)&(w.T>1e-300), w*np.log(np.clip(w,1e-300,None)/np.clip(w.T,1e-300,None)),0.0)))
    JS=0.0
    for i in range(m):
        a=T[i,:]; bb=(p*T[:,i])/p[i]; mm=(a+bb)/2
        JS+=0.5*p[i]*(kl(a,mm)+kl(bb,mm))
    return DS,JS,p
def cur_stats(T,h=1e-4):
    m=T.shape[0]
    def mu(z):
        M=np.zeros((m,m))
        for i in range(m):
            M[i,(i+1)%m]=T[i,(i+1)%m]*math.exp(z)
            M[i,(i-1)%m]=T[i,(i-1)%m]*math.exp(-z)
        return max(np.real(np.linalg.eigvals(M)))
    m0=mu(0); mp=mu(h); mm=mu(-h)
    J=(mp-mm)/(2*h); D=0.5*(mp-2*m0+mm)/(h*h)
    return J,D
def ring_uniform(m,r):
    q=r/(1+r); T=np.zeros((m,m))
    for i in range(m): T[i,(i+1)%m]=q; T[i,(i-1)%m]=1-q
    return T
def ring_nonuniform(m,amps):
    # 顺时针速率 amps[i]，逆时针 1/amps[i]，归一化
    T=np.zeros((m,m))
    for i in range(m):
        T[i,(i+1)%m]=amps[i%len(amps)]
        T[i,(i-1)%m]=1.0/amps[i%len(amps)]
    return T/T.sum(axis=1,keepdims=True)
print("="*94)
print("换族重测：本条 4JS vs TUR J^2/D（比值越大越紧）")
print("="*94)
print("  %-22s %6s %12s %12s %10s %10s %8s" % ("族","r","sigma","4JS","TUR","本条/sig","更紧"))
rows=[]
for m in (3,4,5):
    for r in (1.5,3.0,10.0):
        T=ring_uniform(m,r); DS,JS,p=quant(T); J,D=cur_stats(T)
        mine=4*JS; tur=J*J/D if D>1e-14 else float('nan')
        tag="本条" if mine>tur else "TUR"
        rows.append((m,r,DS,mine,tur,mine/DS,tur/DS,tag))
        print("  %-22s %6.2f %12.6f %12.6f %10.6f %10.4f %10.4f %8s" % ("uniform m=%d"%m,r,DS,mine,tur,mine/DS,tur/DS,tag))
print()
print("  --- 非均匀势（顺时针速率按 (2,1,0.5,...) 变）---")
for m in (3,4,5):
    amps=[2.0**((i%3)-1) for i in range(m)]
    Tend=ring_nonuniform(m,amps)
    for scale in (2.0,10.0):
        # 用整体倾斜放大驱动：顺时针乘 scale，逆时针除 scale
        T=Tend.copy()
        for i in range(m):
            T[i,(i+1)%m]*=scale; T[i,(i-1)%m]/=scale
        T=T/T.sum(axis=1,keepdims=True)
        DS,JS,p=quant(T); J,D=cur_stats(T)
        mine=4*JS; tur=J*J/D if D>1e-14 else float('nan')
        tag="本条" if mine>tur else "TUR"
        rows.append((m,scale,DS,mine,tur,mine/DS,tur/DS,tag))
        print("  %-22s %6.2f %12.6f %12.6f %10.6f %10.4f %10.4f %8s" % ("nonunif m=%d"%m,scale,DS,mine,tur,mine/DS,tur/DS,tag))
print()
wins=sum(1 for r in rows if r[7]=="本条")
print("  总计 %d 例，本条胜 %d 例，TUR 胜 %d 例" % (len(rows),wins,len(rows)-wins))
