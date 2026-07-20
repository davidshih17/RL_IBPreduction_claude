#!/usr/bin/env python
"""Detect substitution cycles in a banked worker-result cache dir.
Builds the directed graph  entry -> {cached integrals appearing in its RHS}
and runs an iterative DFS cycle detection. Prints every cycle found.

Usage: check_cache_cycles.py <work/results dir>
"""
import sys, glob, pickle

d = sys.argv[1]
cache = {}
for f in glob.glob(d + "/*.pkl"):
    try:
        r = pickle.load(open(f, "rb"))
    except Exception as e:
        print(f"unreadable {f}: {e}")
        continue
    integ = r.get("original_integral") or r.get("integral")
    sub = r.get("final_expr") or r.get("substitution") or r.get("result")
    if integ is None or sub is None:
        print(f"unrecognized payload keys in {f}: {list(r)[:8]}")
        continue
    if not r.get("success", True):
        continue
    cache[tuple(integ)] = {tuple(k): v for k, v in sub.items()}

print(f"cache entries: {len(cache)}")
# edges only to integrals that are themselves cache keys
graph = {k: [j for j in v if j in cache and v[j] % 1009] for k, v in cache.items()}

WHITE, GRAY, BLACK = 0, 1, 2
color = {k: WHITE for k in graph}
cycles = []
for start in graph:
    if color[start] != WHITE:
        continue
    stack = [(start, iter(graph[start]))]
    color[start] = GRAY
    path = [start]
    while stack:
        node, it = stack[-1]
        adv = False
        for nxt in it:
            if color[nxt] == WHITE:
                color[nxt] = GRAY
                stack.append((nxt, iter(graph[nxt])))
                path.append(nxt)
                adv = True
                break
            if color[nxt] == GRAY:
                i = path.index(nxt)
                cycles.append(path[i:] + [nxt])
        if not adv:
            color[node] = BLACK
            stack.pop()
            path.pop()

print(f"cycles found: {len(cycles)}")
for c in cycles[:10]:
    print("CYCLE: " + " -> ".join(str(list(t)) for t in c))
