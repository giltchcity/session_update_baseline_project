"""Undefined names by the compiler's own scope analysis (symtable): a name a scope reads that resolves to the module scope (not local,
not a closure variable) but is neither bound at module level nor a builtin. Closures and comprehensions are handled by symtable."""
import builtins, glob, symtable, sys
root = sys.argv[1]
B = set(dir(builtins)) | {'__file__', '__name__', '__doc__', '__spec__', '__package__', '__builtins__', '__loader__', '__path__'}
hits = []
for path in sorted(glob.glob(f'{root}/**/*.py', recursive=True)):
    src = open(path).read()
    try:
        top = symtable.symtable(src, path, 'exec')
    except SyntaxError as e:
        hits.append((path, 'SYNTAX', str(e))); continue
    module_bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    star = 'import *' in src
    def walk(t):
        for s in t.get_symbols():
            if t.get_type() != 'module' and s.is_referenced() and s.is_global() and not s.is_declared_global():
                n = s.get_name()
                if n not in module_bound and n not in B and not star:
                    hits.append((path, t.get_name(), n))
            elif t.get_type() != 'module' and s.is_declared_global() and s.is_referenced():
                n = s.get_name()
                if n not in module_bound and n not in B and not s.is_assigned():
                    hits.append((path, t.get_name(), n))
        for c in t.get_children():
            walk(c)
    walk(top)
    if top is not None:
        for s in top.get_symbols():          # module-level reads of never-bound names
            if s.is_referenced() and not (s.is_assigned() or s.is_imported() or s.is_namespace()) and s.get_name() not in B and not star:
                hits.append((path, '<module>', s.get_name()))
for h in hits:
    print(*h)
print(f'UNDEF_SCAN {root}: {len(hits)} hits')
