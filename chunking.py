import re, ast, os, json, math
from collections import Counter, defaultdict, deque
from typing import List, Dict, Tuple, Set

def retrieve_file_list(response: str):
    # Match everything between the first `[` and the matching `]`, including newlines
    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except (SyntaxError, ValueError) as e:
            print(f"Failed to parse list: {e}")
    return None

CALL_RE = re.compile(r'([A-Za-z_$][A-Za-z0-9_$]*)\s*\(') 

def index_repository(files, keep_symbols=True, top_quantile=0.15, min_edge=1.0):
    """Return a repo-level index ready for LLM consumption."""
    # 1) Collect all functions across all files
    all_funcs = []            # list of dicts
    per_file_source = {}      # file -> source text
    for fp in files:
        if not os.path.exists(fp): 
            continue
        src = open(fp, "r", encoding="utf-8").read()
        per_file_source[fp] = src
        for f in extract_functions(src):
            f["file"] = fp
            all_funcs.append(f)

    if not all_funcs:
        return {"summary": {"files": 0, "functions": 0, "chapters": 0}, "functions": [], "chapters": []}
    
    raw_tokens = [tokenize_function_body(f["code"], keep_symbols=keep_symbols) for f in all_funcs]

    for f, toks in zip(all_funcs, raw_tokens):
        f["token_count"] = len(toks)
        f["tokens"] = toks

    name_to_ids = defaultdict(list)
    for i, f in enumerate(all_funcs):
        # map simple name (last segment if 'Class.method')
        base = f["name"].split(".")[-1].lower()
        if base and base != "<anonymous>":
            name_to_ids[base].append(i)

    def internal_callees(snippet: str):
        # best-effort callee names from code
        return {m.group(1).lower() for m in CALL_RE.finditer(snippet)}

    adj = [defaultdict(float) for _ in all_funcs]
    for i, f in enumerate(all_funcs):
        # internal calls
        for callee in internal_callees(f["code"]):
            for j in name_to_ids.get(callee, []):
                if j != i:
                    adj[i][j] += 3.0
                    adj[j][i] += 0.0  # directed preference, but we keep symmetric weights below as needed

    # same-file mild cohesion
    by_file = defaultdict(list)
    for i, f in enumerate(all_funcs):
        by_file[f["file"]].append(i)
    for _, idxs in by_file.items():
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i][j] += 0.5

    # 4) Thresholded connected components = chapters
    comps = _components_from_adj(adj, min_edge=min_edge)

    # 5) Order within each chapter: try topological-ish by call edges; fallback to file/line
    chapters = []
    for comp in comps:
        order = _order_component(all_funcs, comp, adj)
        chapters.append({
            "title": _title_from_tokens([all_funcs[k]["tokens"] for k in comp]),
            "functions": [
                {
                    "id": k,
                    "file": all_funcs[k]["file"],
                    "name": all_funcs[k]["name"],
                    "start_line": all_funcs[k]["start_line"],
                    "end_line": all_funcs[k]["end_line"],
                    "token_count": all_funcs[k]["token_count"]
                } for k in order
            ]
        })

    # 6) Build a lookup so the LLM can fetch code by id quickly
    functions_view = [
        {
            "id": i,
            "file": f["file"],
            "name": f["name"],
            "start_line": f["start_line"],
            "end_line": f["end_line"],
            "token_count": f["token_count"]
        } for i, f in enumerate(all_funcs)
    ]
    lookup = {i: {"code": f["code"], "file": f["file"], "name": f["name"]} for i, f in enumerate(all_funcs)}

    return {
        "summary": {
            "files": len(per_file_source),
            "functions": len(all_funcs),
            "chapters": len(chapters)
        },
        "functions": functions_view,   # light index (for itinerary table)
        "chapters": chapters,          # chapter itinerary (ordered)
        "lookup": lookup               # id -> code/name/file for chapter rendering
    }

# ---------- helpers ----------

def _components_from_adj(adj, min_edge=1.0):
    n = len(adj)
    seen = [False]*n
    comps = []
    for s in range(n):
        if seen[s]: continue
        q = deque([s]); seen[s] = True; comp=[s]
        while q:
            u = q.popleft()
            for v, w in adj[u].items():
                if w >= min_edge and not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        comps.append(sorted(comp))
    return comps

def _order_component(funcs, comp, adj):
    # prefer call-edge driven order: i -> j if strong call weight
    indeg = {i: 0 for i in comp}
    edges = defaultdict(set)
    for i in comp:
        for j, w in adj[i].items():
            if j in comp and w >= 2.5:
                edges[i].add(j)
    for i in comp:
        for j in edges[i]:
            indeg[j] += 1
    q = deque([i for i in comp if indeg[i]==0])
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in edges[u]:
            indeg[v] -= 1
            if indeg[v]==0:
                q.append(v)
    if len(out) != len(comp):
        # fallback: file then line
        return sorted(comp, key=lambda k: (funcs[k]["file"], funcs[k]["start_line"]))
    return out

def _title_from_tokens(token_lists, k_top=3):
    # tiny, adaptive title: top tokens across the chapter (after boring filter)
    bag = Counter()
    for tl in token_lists:
        bag.update(tl)
    return " ".join([t for t,_ in bag.most_common(k_top)]) or "chapter"

FUNC_PATTERNS = [
    r"function\s+\w+\s*\([^)]*\)\s*{(?:[^{}]|\{[^{}]*\})*}",              # function foo(...) { ...{...}... }
    r"const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*{(?:[^{}]|\{[^{}]*\})*}",       # const foo = (...) => { ... }
    r"\w+\s*\([^)]*\)\s*{(?:[^{}]|\{[^{}]*\})*}"                         # methodName(...) { ... }
]
# Note: patterns allow one-level nested braces in bodies. Good enough for most UI code.
# For very complex nesting, prefer Node-TS parser subprocess later.

def extract_functions(code: str) -> List[Dict]:
    funcs = []
    for pat in FUNC_PATTERNS:
        for m in re.finditer(pat, code, flags=re.DOTALL):
            snippet = m.group(0)
            name = guess_name(snippet)
            start_line = code.count("\n", 0, m.start()) + 1
            end_line   = code.count("\n", 0, m.end()) + 1
            funcs.append({
                "name": name,
                "start_line": start_line,
                "end_line": end_line,
                "code": snippet
            })
    # de-dup (overlaps can happen if patterns match the same thing)
    seen = set()
    uniq = []
    for f in funcs:
        key = (f["start_line"], f["end_line"])
        if key not in seen:
            uniq.append(f); seen.add(key)
    return uniq

def guess_name(snippet: str) -> str:
    # 1) export/default/async/generator function declarations
    m = re.search(r"\bexport\s+(?:default\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", snippet)
    if m: return m.group(1)
    m = re.search(r"\basync\s+function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", snippet)
    if m: return m.group(1)
    m = re.search(r"\bfunction\*?\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", snippet)
    if m: return m.group(1)

    # 2) const/let/var assignments: arrow or function expression (with optional type annotations)
    m = re.search(r"\b(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(", snippet)
    if m: return m.group(1)
    m = re.search(r"\b(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*function\*?\s*\(", snippet)
    if m: return m.group(1)

    # 3) prototype assignments: Foo.prototype.bar = function(...)
    m = re.search(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\.prototype\.([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*function", snippet)
    if m: return f"{m.group(1)}.{m.group(2)}"

    # 4) object literal methods: bar: function(...) or bar: (...) => { }
    m = re.search(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*function\*?\s*\(", snippet)
    if m: return m.group(1)
    m = re.search(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*(?:async\s*)?\([^)]*\)\s*=>", snippet)
    if m: return m.group(1)

    # 5) bare method signature (e.g., within class/object): name(...) { ... }
    #    guard against ending up naming a function keywords like 'if', 'for', etc.
    m = re.search(r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", snippet)
    if m:
        cand = m.group(1)
        js_keywords = {
            "if","for","while","switch","catch","with","return","function","const","let","var",
            "new","class","extends","super","try","else","do","case","default","import","export",
            "from","typeof","instanceof","in","of","await","async","yield","delete","void","this"
        }
        if cand not in js_keywords:
            return cand

    # 6) export default (anonymous)
    if re.search(r"\bexport\s+default\s+function\b", snippet):
        return "default_export"

    # last resort: anonymous with location hint if caller adds it later
    return "<anonymous>"

# tokenization starts here

_RE_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
_RE_LINE_COMMENT  = re.compile(r"//[^\n]*")
_RE_STRING        = re.compile(r"""('([^'\\]|\\.)*'|"([^"\\]|\\.)*"|`([^`\\]|\\.)*`)""", re.DOTALL)

def _strip_comments_and_strings(code: str) -> str:
    code = _RE_BLOCK_COMMENT.sub("", code)
    code = _RE_LINE_COMMENT.sub("", code)
    code = _RE_STRING.sub("", code)
    return code

# split identifiers: fooBar_BAZ9 -> ["foo","bar","baz","9"]
_RE_WORDISH = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+(?:\.\d+)?")
def _split_identifier(tok: str) -> List[str]:
    if not re.match(r"[A-Za-z_$]", tok):
        return [tok]
    # snake + camel split
    parts = re.split(r"[_$]+", tok)
    out = []
    for p in parts:
        # split camel: "XMLHttpRequest2" -> ["xml","http","request","2"]
        segs = re.findall(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+", p)
        out.extend(segs)
    return [s.lower() for s in out if s]

_RE_SYMBOLS = re.compile(r"==|!=|<=|>=|=>|\+\+|--|&&|\|\||[-+*/%&|^~=<>!?:;.,{}\[\]()]")

def tokenize_function_body(snippet: str, keep_symbols: bool = True) -> List[str]:
    """
    Tokenizes TS/JS function body into:
      - identifier subtokens (camel/snake split, lowercased)
      - numbers
      - (optionally) operators/symbols
    Comments & strings removed first.
    """
    body = _strip_comments_and_strings(snippet)
    tokens = []

    # words & numbers
    for m in _RE_WORDISH.finditer(body):
        tok = m.group(0)
        if re.match(r"[A-Za-z_$]", tok):
            tokens.extend(_split_identifier(tok))
        else:
            tokens.append(tok)  # numbers kept as-is

    if keep_symbols:
        tokens.extend(_RE_SYMBOLS.findall(body))

    return tokens

def build_adaptive_boring_set(tokenized_funcs: List[str], top_quantile: float = 0.15) -> Set[str]:
    df = Counter(tokenized_funcs)
    if not df:
        return set()

    # Sort tokens by frequency (descending)
    sorted_tokens = sorted(df.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_tokens)
    k = max(1, int(top_quantile * n))  # number of tokens to keep

    return {token for token, _ in sorted_tokens[:k]}

def apply_boring_filter(tokens: List[str], boring: Set[str]) -> List[str]:
    if not boring: 
        return tokens
    return [t for t in tokens if t not in boring]