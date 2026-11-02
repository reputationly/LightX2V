import json, subprocess, collections, sys
B = subprocess.check_output(["git","merge-base","HEAD","upstream/main"]).decode().strip()
U = "upstream/main"
files = subprocess.check_output(["git","diff","--name-status","-M",B,U,"--","configs"]).decode().splitlines()
def load(rev, path):
    try:
        return json.loads(subprocess.check_output(["git","show",f"{rev}:{path}"],stderr=subprocess.DEVNULL))
    except Exception:
        return None
def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{p}.{k}" if p else k
            if isinstance(v, dict) and k not in ("lora_configs",):
                out.update(flat(v, kk))
            else:
                out[kk] = v
    return out
removed = collections.Counter(); added = collections.Counter(); pairs = collections.Counter(); examples = {}
for line in files:
    parts = line.split("\t")
    st = parts[0]
    if st.startswith("R"): old, new = parts[1], parts[2]
    elif st == "M": old = new = parts[1]
    else: continue
    if not old.endswith(".json"): continue
    a, b = load(B, old), load(U, new)
    if a is None or b is None: continue
    fa, fb = flat(a), flat(b)
    r = set(fa) - set(fb); ad = set(fb) - set(fa)
    for k in r: removed[k] += 1
    for k in ad: added[k] += 1
    for k in r:
        for j in ad:
            pairs[(k, j)] += 1
            examples.setdefault((k,j), (old, fa[k], fb[j]))
print("=== REMOVED keys (count)"); [print(f"{c:4d} {k}") for k, c in removed.most_common(80)]
print("=== ADDED keys (count)"); [print(f"{c:4d} {k}") for k, c in added.most_common(60)]
