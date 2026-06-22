"""Quick inspector: how big is one shard in RAM, what fields are tensors vs Python objects?"""
import sys, os, time, pickle
import torch
import resource


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux -> MB


def main():
    path = sys.argv[1]
    print(f"File: {path}", flush=True)
    print(f"On-disk size: {os.path.getsize(path)/1e6:.1f} MB", flush=True)
    print(f"Baseline RSS: {rss_mb():.1f} MB", flush=True)

    t0 = time.time()
    d = torch.load(path, weights_only=False)
    print(f"Loaded in {time.time()-t0:.2f}s", flush=True)
    print(f"After-load RSS: {rss_mb():.1f} MB", flush=True)

    print("\nField sizes:", flush=True)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            mb = v.element_size() * v.numel() / 1e6
            print(f"  {k}: tensor {tuple(v.shape)} dtype={v.dtype} -> {mb:.1f} MB", flush=True)
        elif isinstance(v, list):
            sample = v[0] if v else None
            try:
                pickle_bytes = len(pickle.dumps(v))
                pmb = pickle_bytes / 1e6
            except Exception as e:
                pmb = -1
            print(f"  {k}: list len={len(v)} sample_type={type(sample).__name__} -> pickled {pmb:.1f} MB", flush=True)
            if v:
                print(f"    sample[0] = {repr(v[0])[:200]}", flush=True)
        else:
            print(f"  {k}: {type(v).__name__}", flush=True)

    # Try mmap loading
    print("\nNow trying with mmap=True:", flush=True)
    del d
    import gc; gc.collect()
    print(f"RSS after free: {rss_mb():.1f} MB", flush=True)
    t0 = time.time()
    d = torch.load(path, weights_only=False, mmap=True)
    print(f"mmap-load time: {time.time()-t0:.2f}s", flush=True)
    print(f"mmap RSS: {rss_mb():.1f} MB", flush=True)


if __name__ == '__main__':
    main()
