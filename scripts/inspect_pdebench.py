"""List every PDEBench HDF5 file with its top-level keys and shapes."""
import sys, os, h5py
root = sys.argv[1]
for dp, _, fs in sorted(os.walk(root)):
    for f in sorted(fs):
        if not f.endswith(('.h5', '.hdf5')):
            continue
        p = os.path.join(dp, f)
        try:
            with h5py.File(p, 'r') as h:
                keys = list(h.keys())
                desc = []
                for k in keys[:6]:
                    o = h[k]
                    if isinstance(o, h5py.Dataset):
                        desc.append(f"{k}{o.shape}{o.dtype}")
                    else:
                        sub = list(o.keys())
                        desc.append(f"{k}/{{{','.join(f'{s}{o[s].shape}' for s in sub[:4] if isinstance(o[s], h5py.Dataset))}}}")
                print(f"{os.path.relpath(p, root)}\t{os.path.getsize(p)/1e9:.1f}G\tnkeys={len(keys)}\t{' '.join(desc)}")
        except Exception as e:
            print(f"{os.path.relpath(p, root)}\tERR {e}")
