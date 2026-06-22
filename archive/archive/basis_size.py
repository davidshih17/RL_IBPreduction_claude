import pickle, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def basis(path):
    try:
        d = pickle.load(open(path,'rb'))
    except Exception as e:
        return None
    # final reduced expression = full_expr_replay (master_integral -> coeff)
    fe = d.get('full_expr_replay') if isinstance(d, dict) else None
    if fe is None:
        for k in ('final_expr','expr'):
            if isinstance(d,dict) and isinstance(d.get(k),dict): fe=d[k]; break
    return len(fe) if isinstance(fe, dict) else None
for p in sys.argv[1:]:
    name = p.split('/actionselect_exp/')[-1].split('/')[0]
    print(f"  {name:20s} basis(n_masters)={basis(p)}")
