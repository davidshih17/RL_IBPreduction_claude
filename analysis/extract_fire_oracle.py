#!/usr/bin/env python
"""Extract the FIRE reduction table (tableAll[40]) into a machine-usable oracle:

  results/fire_oracle_GR.pkl = {
    'targets':   [15-tuple, ...]                 # the ~2100 amplitude integrals
    'solutions': {target: {master: coeff mod p}} # FIRE's reduction, evaluated at
                                                 # the topology kinematic point
    'skipped':   {target: reason}                # entries with a denominator == 0
                                                 # mod p at the point (unlucky prime)
    'masters':   [15-tuple, ...]                 # FIRE's 68-master basis (G[40,..])
  }

Coefficients are rational functions of d and y. We evaluate them EXACTLY over
GF(1009) at the topology's kinematic point (d=41, y=31): numerator and
denominator polynomials evaluated with Fraction-free modular arithmetic via
eval() on a sanitized expression (Mathematica -> Python: '^'->'**', whitespace
and newlines stripped), each of num/den reduced mod p, division = modular
inverse. Denominator == 0 mod p -> entry recorded under 'skipped'.
"""
import os, re, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
SRC = os.path.join(BASE, "topology_input/fire_tables40_math_All.m")
P = 1009
KIN = {"d": 41, "y": 31}

text = open(SRC).read()

# --- masters ---
m_m = re.search(r"masterAll\[40\]\s*=\s*\{(.*?)\}\s*(?:\n\s*\n|$)", text, re.S)
masters = [tuple(int(x) for x in g.split(","))
           for g in re.findall(r"G\[40,\s*\{([^}]*)\}\]", m_m.group(1))]
print(f"masters: {len(masters)}")

# --- table region ---
t_start = text.index("tableAll[40]")
table_txt = text[t_start:]

# Split into rules: FF[40,{...}] -> RHS   (RHS runs until the next FF[ at depth 0
# or the closing of the table). Simpler robust split: find all rule START offsets.
starts = [m.start() for m in re.finditer(r"FF\[40,\s*\{", table_txt)]
print(f"table rules found: {len(starts)}")

MOD_ENV = {"__builtins__": {}}


def eval_poly(expr):
    """Evaluate a Mathematica polynomial/rational snippet mod p at KIN."""
    py = expr.replace("^", "**").replace("\n", " ")
    return eval(py, MOD_ENV, dict(KIN))


def eval_coeff_string(s):
    """Full rational-coefficient string -> GF(p) value.
    Strategy: convert to a Python expression over Fractions of ints is unsafe for
    huge powers; instead evaluate over GF(p) directly by mapping division to
    modular inverse via a tiny expression rewriter: we evaluate the WHOLE
    expression with Python ints and Fraction to stay exact, then reduce."""
    from fractions import Fraction
    py = s.replace("^", "**").replace("\n", " ")
    env = {k: Fraction(v) for k, v in KIN.items()}
    val = eval(py, MOD_ENV, env)
    fr = Fraction(val)
    den = fr.denominator % P
    if den == 0:
        raise ZeroDivisionError("denominator == 0 mod p")
    return (fr.numerator % P) * pow(den, P - 2, P) % P


solutions = {}
skipped = {}
G_RE = re.compile(r"G\[40,\s*\{([^}]*)\}\]")

for i, st in enumerate(starts):
    end = starts[i + 1] if i + 1 < len(starts) else len(table_txt)
    rule = table_txt[st:end]
    tgt_m = re.match(r"FF\[40,\s*\{([^}]*)\}\]\s*->", rule)
    target = tuple(int(x) for x in tgt_m.group(1).split(","))
    rhs = rule[tgt_m.end():].rstrip().rstrip(",}").strip()
    # replace each G[...] with a symbol gN, evaluate the expression symbolically
    # per master by collecting: coeff of gN = eval(rhs with gN=1, others=0)? That
    # is only valid for LINEAR expressions -- FIRE tables are linear in G ✓.
    gs = []
    def sub_g(m):
        idx = tuple(int(x) for x in m.group(1).split(","))
        if idx not in gs:
            gs.append(idx)
        return f"g{gs.index(idx)}"
    rhs_py = G_RE.sub(sub_g, rhs)
    try:
        from fractions import Fraction
        combo = {}
        base_env = {k: Fraction(v) for k, v in KIN.items()}
        for j, g in enumerate(gs):
            env = dict(base_env)
            for jj in range(len(gs)):
                env[f"g{jj}"] = Fraction(1 if jj == j else 0)
            val = Fraction(eval(rhs_py.replace("^", "**").replace("\n", " "),
                                MOD_ENV, env))
            den = val.denominator % P
            if den == 0:
                raise ZeroDivisionError("denominator == 0 mod p")
            c = (val.numerator % P) * pow(den, P - 2, P) % P
            if c:
                combo[g] = c
        solutions[target] = combo
    except ZeroDivisionError as e:
        skipped[target] = str(e)
    except Exception as e:
        skipped[target] = f"parse/eval: {type(e).__name__}: {e}"
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(starts)} rules processed "
              f"({len(skipped)} skipped)", flush=True)

out = {"targets": [tuple(int(x) for x in re.match(
            r"FF\[40,\s*\{([^}]*)\}\]", table_txt[s:]).group(1).split(","))
        for s in starts],
       "solutions": solutions, "skipped": skipped, "masters": masters}
with open(os.path.join(BASE, "results/fire_oracle_GR.pkl"), "wb") as f:
    pickle.dump(out, f)
print(f"\ntargets: {len(out['targets'])}  solved: {len(solutions)}  "
      f"skipped: {len(skipped)}")
print(f"saved -> results/fire_oracle_GR.pkl")
