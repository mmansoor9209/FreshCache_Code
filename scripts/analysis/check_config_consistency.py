"""Check that every parameter recorded in configs/{freshcache,baselines,experiments}/*.yaml equals the value used by the code.

Each YAML entry names its source as `path::NAME[key]...` (a module-level assignment, evaluated without importing the module)
or `path:~regex` (the first match of a regular expression; the captured groups form the value). The experiment code reads
these values from the cited locations, so this check guarantees that the configs describe exactly what is run.
Usage: python scripts/analysis/check_config_consistency.py [--write]   (--write regenerates the values from the code)"""
import ast, pathlib, re, sys
if any(a in ("-h", "--help") for a in sys.argv[1:]): sys.exit(print(__doc__))
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIRS = ("configs/freshcache", "configs/baselines", "configs/experiments")


def _literal(node):
    """Literal value of an expression; dict(k=v) calls, sets, frozensets and simple arithmetic are supported."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        return {k.arg: _literal(k.value) for k in node.keywords}
    if isinstance(node, ast.Set):
        return sorted(_literal(e) for e in node.elts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "frozenset" and len(node.args) == 1:
        return sorted(_literal(node.args[0]))
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.Div)):
        a, b = _literal(node.left), _literal(node.right)
        return {ast.Mult: lambda: a * b, ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Div: lambda: a / b}[type(node.op)]()
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    return ast.literal_eval(node)


def _assignment(tree, name):
    for st in tree.body:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name) and t.id == name: return st.value
                if isinstance(t, ast.Tuple) and isinstance(st.value, ast.Tuple):
                    for e, v in zip(t.elts, st.value.elts):
                        if isinstance(e, ast.Name) and e.id == name: return v
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == name: return st.value
    raise KeyError(name)


def resolve(source):
    """Current value in the code for a `path::NAME[key]` or `path:~regex` source."""
    if ":~" in source:
        path, rx = source.split(":~", 1)
        m = re.search(rx, (REPO / path).read_text(), re.S)
        if not m: raise KeyError(f"pattern not found in {path}: {rx}")
        g = [float(x) if re.fullmatch(r"-?\d+(\.\d+)?", x) else x for x in m.groups()]
        return g[0] if len(g) == 1 else g
    path, expr = source.split("::", 1)
    name, *keys = re.split(r"\[", expr)
    value = _literal(_assignment(ast.parse((REPO / path).read_text()), name))
    for k in keys:
        value = value[ast.literal_eval(k.rstrip("]"))]
    return value


def _norm(v):
    if isinstance(v, tuple): v = list(v)
    if isinstance(v, list): return [_norm(x) for x in v]
    if isinstance(v, dict): return {str(k): _norm(x) for k, x in v.items()}
    if isinstance(v, float) and v.is_integer(): return int(v)
    return v


def main():
    write = "--write" in sys.argv[1:]; bad = 0; n = 0
    for d in CONFIG_DIRS:
        for p in sorted((REPO / d).glob("*.yaml")):
            doc = yaml.safe_load(p.read_text())
            for name, entry in (doc.get("parameters") or {}).items():
                n += 1
                try:
                    cur = _norm(resolve(entry["source"]))
                except Exception as e:
                    print(f"ERROR {p.relative_to(REPO)} {name}: {e}"); bad += 1; continue
                if write: entry["value"] = cur
                elif _norm(entry.get("value")) != cur:
                    print(f"MISMATCH {p.relative_to(REPO)} {name}: config {entry.get('value')!r} != code {cur!r}"); bad += 1
            if write:
                head = "".join(l for l in p.read_text().splitlines(keepends=True) if l.startswith("#"))
                p.write_text(head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
    print(f"{n} parameters checked, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
