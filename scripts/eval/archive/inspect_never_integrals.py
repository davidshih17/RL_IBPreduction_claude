"""Load replay_state.pkl and dump every NEVER integral verbatim."""
import argparse, pickle


def Lrs(ig):
    L = sum(1 for x in ig[:8] if x > 0)
    r = sum(x for x in ig if x > 0)
    s = -sum(x for x in ig if x < 0)
    return L, r, s


def is_paper_master(ig):
    return all(0 <= x <= 1 for x in ig[:8]) and all(x >= 0 for x in ig[8:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('state_pkl')
    args = p.parse_args()
    st = pickle.load(open(args.state_pkl, 'rb'))
    cache = st['cache']
    expr = st['active_expr']
    log_set = st['log_integrals']

    never = []
    for ig in expr:
        if is_paper_master(ig):
            continue
        if ig in cache:
            continue
        if ig in log_set:
            continue
        never.append(ig)

    print(f'NEVER non-masters: {len(never)}\n')
    print(f'{"L":>2s} {"r":>3s} {"s":>3s}   integral')
    for ig in sorted(never, key=lambda i: (-sum(1 for x in i[:8] if x>0),
                                            -sum(x for x in i if x>0),
                                            -(-sum(x for x in i if x<0)))):
        L, r, s = Lrs(ig)
        print(f'{L:>2d} {r:>3d} {s:>3d}   I{list(ig)}')


if __name__ == '__main__':
    main()
