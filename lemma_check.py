import sys, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
rng=np.random.default_rng(20261121)
def kl(x,y):
    x=np.clip(x,1e-300,None); y=np.clip(y,1e-300,None)
    return float(np.sum(x*np.log(x/y)))
def js(a,b):
    m=(a+b)/2
    return 0.5*(kl(a,m)+kl(b,m))
def hell2(a,b):
    return 0.5*float(np.sum((np.sqrt(a)-np.sqrt(b))**2))
print("="*92)
print("引理 1：(x-y)ln(x/y) >= 4(sqrt x - sqrt y)^2   （等价于 对数平均 <= 算术平均）")
print("="*92)
bad1=0; minr=1e9
for _ in range(400000):
    x=10**rng.uniform(-6,6); y=10**rng.uniform(-6,6)
    L=(x-y)*math.log(x/y); R=4*(math.sqrt(x)-math.sqrt(y))**2
    if R>1e-300:
        r=L/R; minr=min(minr,r)
        if L<R-1e-12*max(1,R): bad1+=1
print("  40 万组随机 (x,y)：违例 %d；LHS/RHS 最小 %.12f" % (bad1,minr))
print()
print("="*92)
print("引理 2：JS(a,b) <= Hellinger^2(a,b)")
print("="*92)
bad2=0; maxr=0; arg=None
for n in (2,3,5,10):
    for _ in range(20000):
        a=rng.dirichlet([1.0]*n); b=rng.dirichlet([1.0]*n)
        J=js(a,b); H=hell2(a,b)
        if H>1e-300:
            r=J/H
            if r>maxr: maxr=r; arg=(n,float(J),float(H))
            if J>H+1e-12: bad2+=1
        # 极端：取单点分布
        for _ in range(200):
            i=rng.integers(n); j=rng.integers(n)
            e1=np.zeros(n); e1[i]=1; e2=np.zeros(n); e2[j]=1
            J=js(e1,e2); H=hell2(e1,e2)
            if H>1e-300:
                r=J/H
                if r>maxr: maxr=r; arg=(n,float(J),float(H))
                if J>H+1e-12: bad2+=1
print("  8 万组随机分布 + 极端单点分布：违例 %d" % bad2)
print("  JS/Hellinger^2 最大值 %.6f  (n=%d, JS=%.4f, H2=%.4f)" % (maxr,arg[0],arg[1],arg[2]))
print()
print("="*92)
print("推论：DeltaS >= 4H(pi) >= 4 JS_row  （H(pi)=sum_{i<j}(sqrt w_ij - sqrt w_ji)^2）")
print("="*92)
print("  若两引理成立，则逐边/逐态求和即得。等号条件：两处都取等 ⟹ x=y ⟹ w_ij=w_ji ⟹ 细致平衡。")
