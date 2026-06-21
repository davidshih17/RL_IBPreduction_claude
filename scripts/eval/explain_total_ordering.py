"""Demonstrate the v8 total ordering on concrete pentagonbox integrals.
weight()/_target_key() are pure functions of the integral tuple (no topology),
so we reproduce them exactly here."""

def weight(i):
    return (sum(max(0, x) for x in i),
            -sum(min(0, x) for x in i),
            tuple(abs(x) for x in i))

def target_key(i):
    w = weight(i)
    return (-w[0], -w[1], w[2])

start = (0, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0)   # the probe-74 start integral
start_key = target_key(start)

# A few other integrals, all with the SAME (r,s)=(7,4) as start, plus one higher.
examples = {
    "start                         ": start,
    "(7,4) idx1: 1->2,  idx7:1->0  ": (0, 2, 1, 1, 1, 1, 1, 0, -4, 0, 0),
    "(7,4) split -4 -> -3,-1       ": (0, 1, 1, 1, 1, 1, 1, 1, -3, -1, 0),
    "(7,4) idx8 |abs| 4->5, idx1->0": (0, 0, 1, 1, 1, 1, 1, 1, -5, 0, 0),
    "(8,4) higher r                ": (1, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0),
}

print(f"start            = {start}")
print(f"weight(start)    = {weight(start)}")
print(f"start_key        = {start_key}")
print(f"abs_tuple(start) = {weight(start)[2]}\n")
print(f"{'label':32s} {'r':>2} {'s':>2}  abs_tuple                     "
      f"key<=start? (active)  vs-start")
for label, i in examples.items():
    w = weight(i)
    k = target_key(i)
    active = k <= start_key
    if k < start_key:
        rel = "HIGHER weight (above start)"
    elif k == start_key:
        rel = "EQUAL (the start itself)"
    else:
        rel = "LOWER weight (sub-weight -> STRIPPED)"
    print(f"{label:32s} {w[0]:>2} {w[1]:>2}  {str(w[2]):28s}  "
          f"{str(active):5s}                 {rel}")
