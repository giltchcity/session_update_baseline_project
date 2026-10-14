"""Rewrite the '── 思考与来源（10-14 补写）──' comment blocks of a source file as '原理：… 来源：…' blocks (comments only).
For every such block: the header line goes; '思考：' becomes '原理：'; the '状态：…。' tags go (classification, not principle);
the text is re-joined (a word cut by the old wrap is re-joined when the joined token is an identifier of the file) and
re-wrapped at the file's indentation. Code lines are never touched; the script checks that the file's AST is unchanged.
usage: reflow_comments.py FILE [FILE ...]
"""
import ast
import re
import sys
import unicodedata

HEADER = re.compile(r'^(\s*)# ── 思考与来源（10-14 补写）──\s*$')
WIDTH = 118


def width(ch):
    return 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1


def wrap(text, indent):
    prefix = ' ' * indent + '# '
    room = WIDTH - len(prefix)
    out, line, w = [], '', 0
    tokens = re.findall(r'[A-Za-z0-9_.:/\-\[\]()=<>+*%\'",]+|\s+|.', text)
    for tok in tokens:
        tw = sum(width(c) for c in tok)
        closing = re.match(r'[，。、；：）”’》？！]', tok) is not None      # never starts a line
        if w + tw > room and line.strip() and not closing:
            out.append(prefix + line.rstrip())
            line, w = ('' if tok.isspace() else tok), (0 if tok.isspace() else tw)
        else:
            line += tok
            w += tw
    if line.strip():
        out.append(prefix + line.rstrip())
    return out


def cjk(ch):
    return unicodedata.east_asian_width(ch) in ('W', 'F') and not re.match(r'[，。、；：）（“”‘’《》？！]', ch)


def join(parts, idents):
    """re-join the old wrap: a cut identifier or number is re-joined, a CJK/ASCII boundary keeps its space"""
    text = ''
    for p in parts:
        if not text:
            text = p
            continue
        a, b = text[-1], p[0]
        if re.match(r'[A-Za-z0-9_]', a) and re.match(r'[A-Za-z0-9_]', b):
            left = re.search(r'[A-Za-z0-9_]+$', text).group(0)
            right = re.match(r'[A-Za-z0-9_]+', p).group(0)
            glued = (left.isdigit() and right[0].isdigit()) or (left + right) in idents
            text += p if glued else ' ' + p
        elif (cjk(a) and re.match(r'[A-Za-z0-9_(\[]', b)) or (re.match(r'[A-Za-z0-9_)\]]', a) and cjk(b)):
            text += ' ' + p
        elif re.match(r'[A-Za-z0-9_)\]×]', a) and re.match(r'[A-Za-z0-9_(]', b):
            text += ' ' + p
        else:
            text += p
    return text


for path in sys.argv[1:]:
    src = open(path).read()
    tree0 = ast.dump(ast.parse(src))
    idents = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', src))
    lines = src.splitlines()
    out, i, n = [], 0, 0
    while i < len(lines):
        m = HEADER.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = len(m.group(1))
        j = i + 1
        parts = []
        while j < len(lines) and lines[j].lstrip().startswith('#') and not HEADER.match(lines[j]):
            parts.append(lines[j].strip()[1:].strip())
            j += 1
        text = join([p for p in parts if p], idents)
        text = text.replace('思考：', '原理：', 1)
        text = re.sub(r'状态：[^。]*。', '', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        out.extend(wrap(text, indent))
        n += 1
        i = j
    new = '\n'.join(out) + ('\n' if src.endswith('\n') else '')
    assert ast.dump(ast.parse(new)) == tree0, f'{path}: AST changed'
    open(path, 'w').write(new)
    print(f'{path}: {n} blocks rewritten')
