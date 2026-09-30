"""公開ノートから埋め込み main.py を取り出す (ノートのコードは実行せず、ast で定数だけ復号する)。

  .venv/bin/kaggle kernels pull <user>/<slug> -p tmp/nb/<slug> -m
  .venv/bin/python tests/nb_extract.py tmp/nb/<slug> agents/pub_<tag>      # main.py (+ LICENSE/NOTICE があれば) を書き出し sha256 を表示

対応する埋め込み形: FILES={'main.py': b85+zlib} / AGENT_B64 (gzip+b64) / ARCHIVE_B85 (tar.gz) / ARCHIVE_PARTS.append(...) (tar.gz) / %%writefile。
EXPECTED_MAIN_SHA256 等があれば照合する。既存の agents/pub_* ・ third_party/public_agents/* と同一ハッシュなら再掲と報告する。
"""
import ast, base64, glob, gzip, hashlib, io, json, lzma, os, sys, tarfile, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def literals(src):
    out = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:  # `!shell` セル
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                value = node.value
                # Explicit byte-literal concatenation only, never execute a notebook call.
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                        and value.func.attr == 'join' and isinstance(value.func.value, ast.Constant)
                        and value.func.value.value == b'' and len(value.args) == 1 and not value.keywords):
                    try:
                        chunks = ast.literal_eval(value.args[0])
                    except (ValueError, TypeError, SyntaxError):
                        continue
                    if isinstance(chunks, (list, tuple)) and all(isinstance(x, bytes) for x in chunks):
                        out[node.targets[0].id] = b''.join(chunks)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "attr", "") == "append" \
                and getattr(node.value.func.value, "id", "") == "ARCHIVE_PARTS":
            out.setdefault("ARCHIVE_PARTS", []).append(ast.literal_eval(node.value.args[0]))
    return out


def tar_files(b):
    with tarfile.open(fileobj=io.BytesIO(b), mode="r:gz") as tf:
        return {n: tf.extractfile(n).read() for n in tf.getnames() if tf.extractfile(n)}


def extract(nb_path):
    """返り値: ({filename: bytes}, expected_main_sha or None)"""
    cells = [''.join(c["source"]) for c in json.load(open(nb_path))["cells"] if c["cell_type"] == "code"]
    parts, expect = [], None
    for src in cells:
        L = literals(src)
        expect = expect or L.get("EXPECTED_MAIN_SHA256") or (L.get("EXPECTED") or {}).get("main.py")
        if isinstance(L.get('SOURCE_BYTES'), bytes):
            return {'main.py': L['SOURCE_BYTES']}, expect
        if "PAYLOAD_B85" in L:
            blob = lzma.decompress(base64.b85decode(L["PAYLOAD_B85"]))
            hashes = L.get("EXPECTED_SHA256")
            if isinstance(hashes, dict):
                stream = io.BytesIO(blob)
                def take(n):
                    data = stream.read(n)
                    if len(data) != n:
                        raise ValueError("truncated payload bundle")
                    return data
                files = {}
                for _ in range(int.from_bytes(take(2), "big")):
                    name = take(int.from_bytes(take(2), "big")).decode()
                    data = take(int.from_bytes(take(8), "big"))
                    if name in files or hashlib.sha256(data).hexdigest() != hashes.get(name):
                        raise ValueError(f"duplicate file or SHA mismatch: {name}")
                    files[name] = data
                if stream.read(1):
                    raise ValueError("trailing payload bundle data")
                return files, hashes.get("main.py")
            return {"main.py": blob}, expect
        if "FILES" in L:
            return {k: zlib.decompress(base64.b85decode(v)) for k, v in L["FILES"].items()}, expect
        if "AGENT_B64" in L:
            return {"main.py": gzip.decompress(base64.b64decode(L["AGENT_B64"]))}, expect
        if "ARCHIVE_B85" in L:
            return tar_files(base64.b85decode(''.join(L["ARCHIVE_B85"].split()).encode())), expect
        parts += L.get("ARCHIVE_PARTS", [])
        if src.startswith("%%writefile") and "main.py" in src.split("\n", 1)[0]:
            return {"main.py": src.split("\n", 1)[1].encode()}, expect
    if parts:
        return tar_files(base64.b85decode(''.join(parts).encode())), expect
    raise SystemExit("no known embedding found")


def main():
    src, out = sys.argv[1], sys.argv[2]
    nb = src if src.endswith(".ipynb") else glob.glob(os.path.join(src, "*.ipynb"))[0]
    files, expect = extract(nb)
    main_sha = hashlib.sha256(files["main.py"]).hexdigest()
    if expect and expect != main_sha:
        raise SystemExit("SHA MISMATCH — refusing to write extracted files")
    known = {hashlib.sha256(open(p, "rb").read()).hexdigest(): p for p in glob.glob(f"{ROOT}/agents/pub_*/main.py") + glob.glob(f"{ROOT}/third_party/public_agents/*/main.py")}
    if main_sha in known:
        print(f"same bytes as {os.path.relpath(known[main_sha], ROOT)} (sha {main_sha[:12]}) — 再掲。書き出さない"); return
    os.makedirs(out, exist_ok=True)
    for name, data in files.items():
        if name in ("main.py", "base_agent.py", "shop_predictor.py", "shop_overlay.py", "LICENSE.txt", "NOTICE.txt"):
            open(os.path.join(out, name), "wb").write(data)
    print(f"{out}/main.py {len(files['main.py'])} B sha {main_sha[:12]}", "sha OK" if expect == main_sha else ("SHA MISMATCH" if expect else "(no expected sha in notebook)"))
    last = [l for l in files["main.py"].decode(errors="replace").split("\n") if l.startswith("agent") and "=" in l]
    print("last agent binding:", last[-1][:100] if last else "none — entry を確認")


if __name__ == "__main__":
    main()
