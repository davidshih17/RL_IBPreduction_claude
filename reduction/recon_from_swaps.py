#!/usr/bin/env python
"""Reconstruct a placeholder's TRUE loop relabeling from its clean SWAPS (not the literal
ing, whose 'fixed' entries can be fake). A prop that maps to a DIFFERENT prop is reliable;
'fixed' props may actually map to combinations. Then derive the full momentum map."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
PM={0:(1,0,0,0,0,0),1:(1,0,1,0,0,0),2:(1,0,1,1,0,0),3:(1,0,1,1,1,0),
    4:(0,1,0,0,0,0),5:(0,1,1,1,1,0),6:(0,1,1,1,1,1),7:(1,-1,0,0,0,0),
    8:(1,0,1,1,1,1),9:(0,1,1,0,0,0),10:(0,1,1,1,0,0)}
def vadd(a,b): return tuple(x+y for x,y in zip(a,b))
def vsub(a,b): return tuple(x-y for x,y in zip(a,b))
def vscale(s,a): return tuple(s*x for x in a)
def ppart(v): return (0,0)+v[2:]
NAMES=['k1','k2','p1','p2','p3','p4']
def vstr(v):
    s=""
    for i,x in enumerate(v):
        if x==0: continue
        t=NAMES[i] if abs(x)==1 else f"{abs(x)}*{NAMES[i]}"
        s+=("-" if x<0 else ("+" if s else ""))+t
    return s or "0"
def match(mg):
    for h in range(11):
        if mg==PM[h] or mg==vscale(-1,PM[h]): return h
    return None
def reconstruct(ing, present):
    R1c={PM[0]}; R2c={PM[4]}
    for g in present:
        sg=ing[g]
        if sg==-1: continue
        a,b=PM[g][:2]
        for eps in (1,-1):
            rhs=vsub(vscale(eps,PM[sg]),ppart(PM[g]))
            if (a,b)==(1,0): R1c.add(rhs)
            elif (a,b)==(0,1): R2c.add(rhs)
    best=None
    for R1 in R1c:
        for R2 in R2c:
            ok=True; nclean=0; nswap=0
            for g in present:
                sg=ing[g]
                if sg==-1: continue
                a,b=PM[g][:2]
                mg=vadd(vadd(vscale(a,R1),vscale(b,R2)),ppart(PM[g]))
                h=match(mg)
                if h is not None:
                    nclean+=1
                    if h!=sg: ok=False; break           # clean map disagrees with ing
                    if sg!=g: nswap+=1
                else:
                    if sg!=g: ok=False; break            # ing claimed a swap but got a combination
            if ok and nclean>0 and (best is None or (nswap,nclean)>(best[2],best[3])):
                best=(R1,R2,nswap,nclean)
    if best is None: return None
    return [('k1',vstr(best[0])),('k2',vstr(best[1]))]

if __name__=="__main__":
    from sailir.symmetries import parse_symmetries
    from symmetry_engine import N, derive_transform
    TA=os.path.join(BASE,"results/kira_reduce_161/sectormappings/TA")
    recs=parse_symmetries(os.path.join(TA,"sectorSymmetries"),N,2)+parse_symmetries(os.path.join(TA,"sectorRelations"),N,2)
    def props(sec): return [g for g in range(8) if sec>>g&1]
    # self-check: momentum-map records reconstruct to an equivalent relabeling (same induced clean map)
    mm_ok=mm_bad=0
    for r in recs:
        ls=list(r.loop_substs)
        if any(rhs=="placeholder" for _,rhs in ls): continue
        rec=reconstruct(r.ing, props(r.source_sector))
        mm_ok+= 1 if rec else 0; mm_bad+= 0 if rec else 1
    print(f"momentum-map self-check: {mm_ok} reconstructed, {mm_bad} failed")
    ph_ok=ph_bad=0; example=None
    for r in recs:
        if not any(rhs=="placeholder" for _,rhs in list(r.loop_substs)): continue
        rec=reconstruct(r.ing, props(r.source_sector))
        if rec: 
            ph_ok+=1
            if example is None and r.source_sector==55: example=(r,rec)
        else: ph_bad+=1
    print(f"placeholder reconstruction FROM SWAPS: {ph_ok} succeeded, {ph_bad} failed")
    # show the sector-55 example + its full M
    if example is None:
        for r in recs:
            if any(rhs=="placeholder" for _,rhs in list(r.loop_substs)):
                rr=reconstruct(r.ing,props(r.source_sector))
                if rr: example=(r,rr); break
    r,rec=example
    print(f"\nexample placeholder sector {r.source_sector}, ing={list(r.ing)[:8]}")
    print(f"  reconstructed relabeling: k1->{rec[0][1]}, k2->{rec[1][1]}")
    Mf,cf=derive_transform(rec)
    nm=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11']
    for i in range(N):
        terms=" + ".join(f"{v}*{nm[j]}" for j,v in sorted(Mf[i].items()))
        print(f"    {nm[i]:4s}-> {terms}{'  [combination]' if len(Mf[i])>1 else ''}")
