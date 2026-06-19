import pickle, sys, os
R=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,'scripts','eval'))
d=pickle.load(open(sys.argv[1],'rb'))
print("type:",type(d).__name__)
if isinstance(d,dict):
    for k,v in d.items():
        info=f"len={len(v)}" if hasattr(v,'__len__') else f"={v}"
        print(f"  {k}: {type(v).__name__} {info}")
    # best_state
    bs=d.get('best_state')
    if bs is not None:
        print("  best_state fields:", getattr(bs,'_fields',None))
        print("  best_state.n_non_masters:", getattr(bs,'n_non_masters',None))
    fe=d.get('full_expr_replay')
    if isinstance(fe,dict):
        from sailir.ibp_env import is_master
        ms=[k for k in fe if is_master(k)]
        print(f"  full_expr_replay: {len(fe)} terms, {len(ms)} masters, {len(fe)-len(ms)} non-masters")
