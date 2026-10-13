"""Impact (t = 0, 1) effects of the surprise jump, embodied progress: Ramsey vs annual OLG, log and habits.
Run: python3 jump_impacts.py > jump_impacts.txt"""
import sys,json; sys.path.insert(0,'.')
import numpy as np, olg_dynamics as od
rv=od.rv; r0=od.r0
j=json.load(open('results_emb.json')); cK,cA=j['cK'],j['cA']; T=250
print("model, block, cap | r(t=0->1) | r1 yr2 | lnC0 | lnGDP0 | dI/GDP0 | lnW0 | labsh0 | lnGDP1 | lnC1")
for name,h in od.BLOCKS:
    olg=od.OLGH(od.E,h)
    for cap in "KA":
        hh=rv.HH(1,h,0,name)
        rr=rv.run("jump",cap,"emb",hh,cA,cK=cK,T=T)
        m=od.OLGModel("jump",cap,od.olg_size(olg,cap),olg,cK,cA,T)
        oo=m.solve(rr['z'])
        for mdl,res in (("Ramsey",rr),("OLG",oo)):
            print(f"{mdl:6s} {name:13s} {cap} | {100*res['r'][0]:6.2f} | {100*res['r'][1]:6.2f} | {100*res['lnC'][0]:6.2f} | {100*res['lnYg'][0]:5.2f} | {100*(res['invg'][0]-res['invg0']):+5.2f} | {100*res['lnW'][0]:5.2f} | {100*res['labsh'][0]:5.2f} | {100*res['lnYg'][1]:5.2f} | {100*res['lnC'][1]:6.2f}")
