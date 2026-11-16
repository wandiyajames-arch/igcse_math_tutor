"""Safe parser for typed answers (no eval, no sympify, no SymPy import).

Turns a student's text into an immutable AST that keeps exactly what was
written (brackets, operator styles, term order, number text), so later form
checks can see e.g. that ``2a+3b+5a-9b`` is not simplified.

Ambiguous notation abstains instead of being guessed (``1/2x``, ``x^2y``,
``1,5``, ``2 1/3``, ``√2x``). ``parse()`` never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

from .verdict import Reason, Verdict, abstain

MAX_CHARS = 200
MAX_TOKENS = 100
MAX_DEPTH = 10
MAX_DIGITS = 15
MAX_EXPONENT = 20
MAX_ITEMS = 10

UNITS = ("mm", "cm", "km", "ml", "kg", "min", "m", "l", "g", "s", "h")
UNIT_POWERS = ("²", "³", "^2", "^3")
CURRENCIES = ("$", "£", "€")

_NORMALISE = {
    "−": "-", "–": "-", "—": "-", "‐": "-",
    "×": "*", "·": "*", "∙": "*", "÷": "/",
    "≤": "<=", "≥": ">=", "⩽": "<=", "⩾": ">=",
    "（": "(", "）": ")", "\t": " ", " ": " ",
}
_ALLOWED = set("0123456789.+-*/^()=<>,;:%°√π²³ $£€") | set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ----------------------------------------------------------------- AST nodes
@dataclass(frozen=True)
class Number:
    text: str

    @property
    def value(self) -> Fraction:
        return Fraction(self.text)


@dataclass(frozen=True)
class Variable:
    name: str


@dataclass(frozen=True)
class Pi:
    pass


@dataclass(frozen=True)
class Sqrt:
    arg: "Node"
    bracketed: bool


@dataclass(frozen=True)
class Group:
    inner: "Node"


@dataclass(frozen=True)
class Neg:
    operand: "Node"


@dataclass(frozen=True)
class Pos:
    operand: "Node"


@dataclass(frozen=True)
class Add:
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Sub:
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Mul:
    left: "Node"
    right: "Node"
    style: str  # "explicit" (*, ×) or "implicit"


@dataclass(frozen=True)
class Div:
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Pow:
    base: "Node"
    exp: "Node"
    style: str  # "caret" or "super"


@dataclass(frozen=True)
class Tuple:
    items: tuple


@dataclass(frozen=True)
class Quantity:
    value: "Node"
    unit: Optional[str]
    currency: Optional[str]


@dataclass(frozen=True)
class Relation:
    left: "Node"
    op: str
    right: "Node"


@dataclass(frozen=True)
class Ratio:
    parts: tuple


@dataclass(frozen=True)
class Sequence:
    items: tuple


Node = Union[Number, Variable, Pi, Sqrt, Group, Neg, Pos, Add, Sub, Mul, Div,
             Pow, Tuple, Quantity, Relation, Ratio, Sequence]
EXPRESSION_NODES = (Number, Variable, Pi, Sqrt, Group, Neg, Pos, Add, Sub, Mul, Div, Pow)


@dataclass(frozen=True)
class ParseResult:
    node: Optional[Node] = None
    verdict: Optional[Verdict] = None

    @property
    def ok(self) -> bool:
        return self.node is not None


class _Abstain(Exception):
    def __init__(self, reason: Reason, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


# ----------------------------------------------------------------- tokenizer
@dataclass(frozen=True)
class Tok:
    kind: str   # NUM VAR PI SQRT OP SUP LP RP SEP COLON REL UNIT CUR
    text: str
    space_before: bool


def _normalise(text: str) -> str:
    out = []
    for ch in text:
        out.append(_NORMALISE.get(ch, ch))
    s = "".join(out).strip()
    s = s.replace("sqrt", "√")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _dig(c: str) -> bool:
    return c in "0123456789"


def _tokenize(s: str, variables: frozenset[str]) -> list[Tok]:
    for ch in s:
        if ch not in _ALLOWED:
            raise _Abstain(Reason.UNPARSEABLE, f"character not allowed: {ch!r}")
    toks: list[Tok] = []
    i, n = 0, len(s)
    space = False
    while i < n:
        ch = s[i]
        if ch == " ":
            space = True
            i += 1
            continue
        if _dig(ch) or (ch == "." and i + 1 < n and _dig(s[i + 1])):
            j = i
            while j < n and _dig(s[j]):
                j += 1
            if j < n and s[j] == ".":
                j += 1
                if j >= n or not _dig(s[j]):
                    raise _Abstain(Reason.UNPARSEABLE, "number ends with a point")
                while j < n and _dig(s[j]):
                    j += 1
                if j < n and s[j] == ".":
                    raise _Abstain(Reason.UNPARSEABLE, "two decimal points")
            text = s[i:j]
            if sum(_dig(c) for c in text) > MAX_DIGITS:
                raise _Abstain(Reason.TOO_COMPLEX, "number has too many digits")
            toks.append(Tok("NUM", text, space))
            i = j
        elif ch.isalpha() and ch != "π":
            j = i
            while j < n and s[j].isalpha() and s[j] != "π":
                j += 1
            toks.extend(_word_tokens(s[i:j], space))
            i = j
        elif ch == "π":
            toks.append(Tok("PI", "π", space)); i += 1
        elif ch == "√":
            toks.append(Tok("SQRT", "√", space)); i += 1
        elif ch in "+-*/":
            toks.append(Tok("OP", ch, space)); i += 1
        elif ch == "^":
            toks.append(Tok("OP", "^", space)); i += 1
        elif ch in "²³":
            toks.append(Tok("SUP", ch, space)); i += 1
        elif ch == "(":
            toks.append(Tok("LP", ch, space)); i += 1
        elif ch == ")":
            toks.append(Tok("RP", ch, space)); i += 1
        elif ch in ",;":
            toks.append(Tok("SEP", ch, space)); i += 1
        elif ch == ":":
            toks.append(Tok("COLON", ch, space)); i += 1
        elif ch in "<>=":
            if ch in "<>" and i + 1 < n and s[i + 1] == "=":
                toks.append(Tok("REL", ch + "=", space)); i += 2
            else:
                toks.append(Tok("REL", ch, space)); i += 1
        elif ch in "%°":
            toks.append(Tok("UNIT", ch, space)); i += 1
        elif ch in CURRENCIES:
            toks.append(Tok("CUR", ch, space)); i += 1
        else:  # pragma: no cover - guarded by _ALLOWED
            raise _Abstain(Reason.UNPARSEABLE, f"unexpected {ch!r}")
        space = False
    if len(toks) > MAX_TOKENS:
        raise _Abstain(Reason.TOO_COMPLEX, "too many tokens")
    return _resolve_units(toks, variables)


def _word_tokens(word: str, space: bool) -> list[Tok]:
    if word in ("or", "and"):
        return [Tok("SEP", word, space)]
    if word == "pi":
        return [Tok("PI", "π", space)]
    # Keep the whole word for unit resolution; split later if not a unit.
    return [Tok("WORD", word, space)]


def _resolve_units(toks: list[Tok], variables: frozenset[str]) -> list[Tok]:
    """A unit word is accepted only at the end of an answer item, after a value."""
    out: list[Tok] = []
    for idx, t in enumerate(toks):
        if t.kind != "WORD":
            out.append(t)
            continue
        nxt = toks[idx + 1] if idx + 1 < len(toks) else None
        # unit powers like cm² / cm^2
        follow = idx + 1
        if nxt is not None and nxt.kind == "SUP":
            follow = idx + 2
        elif (nxt is not None and nxt.text == "^" and idx + 2 < len(toks)
              and toks[idx + 2].text in ("2", "3")):
            follow = idx + 3
        at_end = follow >= len(toks) or toks[follow].kind in ("SEP", "COLON", "REL", "RP")
        prev = out[-1] if out else None
        after_value = prev is not None and prev.kind in ("NUM", "RP", "PI", "VAR", "SUP")
        if t.text in UNITS and at_end and after_value:
            if all(c in variables for c in t.text):
                raise _Abstain(Reason.AMBIGUOUS_INPUT, f"'{t.text}' could be a unit or a variable")
            out.append(Tok("UNIT", t.text, t.space_before))
            continue
        if len(t.text) > 1 and not all(c in variables for c in t.text):
            raise _Abstain(Reason.UNPARSEABLE, f"unknown word {t.text!r}")
        for k, c in enumerate(t.text):
            out.append(Tok("VAR", c, t.space_before if k == 0 else False))
    # attach unit powers
    merged: list[Tok] = []
    i = 0
    while i < len(out):
        t = out[i]
        if t.kind == "UNIT" and t.text in ("mm", "cm", "m", "km"):
            if i + 1 < len(out) and out[i + 1].kind == "SUP":
                merged.append(Tok("UNIT", t.text + out[i + 1].text, t.space_before)); i += 2; continue
            if (i + 2 < len(out) and out[i + 1].text == "^" and out[i + 2].text in ("2", "3")):
                merged.append(Tok("UNIT", t.text + ("²" if out[i + 2].text == "2" else "³"),
                                  t.space_before)); i += 3; continue
        merged.append(t)
        i += 1
    return merged


# -------------------------------------------------------------------- parser
_IMPLICIT_START = ("VAR", "PI", "SQRT", "LP")


class _Parser:
    def __init__(self, toks: list[Tok]):
        self.t = toks
        self.i = 0
        self.depth = 0

    # helpers
    def peek(self, k: int = 0) -> Optional[Tok]:
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def take(self) -> Tok:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str) -> Tok:
        tok = self.peek()
        if tok is None or tok.kind != kind:
            raise _Abstain(Reason.UNPARSEABLE, f"expected {kind}")
        return self.take()

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _Abstain(Reason.TOO_COMPLEX, "brackets nested too deeply")

    def leave(self) -> None:
        self.depth -= 1

    # grammar
    def parse_answer(self) -> Node:
        items = [self.parse_item()]
        while self.peek() is not None and self.peek().kind == "SEP":
            sep = self.take()
            if sep.text == "," and items:
                prev_tok = self.t[self.i - 2] if self.i >= 2 else None
                nxt = self.peek()
                if (prev_tok is not None and prev_tok.kind == "NUM" and nxt is not None
                        and nxt.kind == "NUM" and not sep.space_before and not nxt.space_before):
                    raise _Abstain(Reason.AMBIGUOUS_INPUT, "comma between digits (decimal comma or thousands?)")
            if self.peek() is None:
                raise _Abstain(Reason.UNPARSEABLE, "separator at the end")
            items.append(self.parse_item())
            if len(items) > MAX_ITEMS:
                raise _Abstain(Reason.TOO_COMPLEX, "too many items")
        if self.peek() is not None:
            tok = self.peek()
            if tok.kind == "NUM":
                raise _Abstain(Reason.AMBIGUOUS_INPUT, "number directly after a value")
            raise _Abstain(Reason.UNPARSEABLE, f"unexpected {tok.text!r}")
        return items[0] if len(items) == 1 else Sequence(tuple(items))

    def parse_item(self) -> Node:
        first = self.parse_relation()
        if self.peek() is not None and self.peek().kind == "COLON":
            parts = [first]
            while self.peek() is not None and self.peek().kind == "COLON":
                self.take()
                parts.append(self.parse_relation())
            if any(isinstance(p, (Relation, Tuple)) for p in parts):
                raise _Abstain(Reason.UNPARSEABLE, "relation or tuple inside a ratio")
            return Ratio(tuple(parts))
        return first

    def parse_relation(self) -> Node:
        left = self.parse_quantity()
        if self.peek() is not None and self.peek().kind == "REL":
            op = self.take().text
            right = self.parse_quantity()
            if self.peek() is not None and self.peek().kind == "REL":
                raise _Abstain(Reason.TOO_COMPLEX, "more than one relation in one item")
            return Relation(left, op, right)
        return left

    def parse_quantity(self) -> Node:
        cur = None
        if self.peek() is not None and self.peek().kind == "CUR":
            cur = self.take().text
        if self.peek() is not None and self.peek().kind == "LP" and self._is_tuple_ahead():
            if cur:
                raise _Abstain(Reason.UNPARSEABLE, "currency on a tuple")
            return self.parse_tuple()
        value = self.parse_expr()
        unit = None
        if self.peek() is not None and self.peek().kind == "UNIT":
            unit = self.take().text
        if cur is None and unit is None:
            return value
        return Quantity(value, unit, cur)

    def _is_tuple_ahead(self) -> bool:
        depth = 0
        j = self.i
        while j < len(self.t):
            k = self.t[j]
            if k.kind == "LP":
                depth += 1
            elif k.kind == "RP":
                depth -= 1
                if depth == 0:
                    nxt = self.t[j + 1] if j + 1 < len(self.t) else None
                    return False if nxt is None or nxt.kind in ("SEP", "COLON", "REL") else False
            elif k.kind == "SEP" and k.text == "," and depth == 1:
                return True
            j += 1
        return False

    def parse_tuple(self) -> Node:
        self.expect("LP")
        self.enter()
        items = []
        if self.peek() is not None and self.peek().kind == "RP":
            raise _Abstain(Reason.UNPARSEABLE, "empty tuple")
        while True:
            items.append(self.parse_expr())
            tok = self.peek()
            if tok is not None and tok.kind == "SEP" and tok.text == ",":
                self.take()
                if self.peek() is not None and self.peek().kind == "RP":
                    raise _Abstain(Reason.UNPARSEABLE, "tuple ends with a comma")
                continue
            break
        self.expect("RP")
        self.leave()
        if len(items) < 2:
            raise _Abstain(Reason.UNPARSEABLE, "tuple needs at least two items")
        if len(items) > MAX_ITEMS:
            raise _Abstain(Reason.TOO_COMPLEX, "too many tuple items")
        nxt = self.peek()
        if nxt is not None and nxt.kind not in ("SEP",):
            raise _Abstain(Reason.UNPARSEABLE, "a tuple must be a whole answer item")
        return Tuple(tuple(items))

    def parse_expr(self) -> Node:
        node = self.parse_term()
        while self.peek() is not None and self.peek().kind == "OP" and self.peek().text in "+-":
            op = self.take().text
            right = self.parse_term()
            node = Add(node, right) if op == "+" else Sub(node, right)
        return node

    def parse_term(self) -> Node:
        node, flag = self.parse_unary()
        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok.kind == "OP" and tok.text in "*/":
                self.take()
                right, rflag = self.parse_unary()
                if tok.text == "*":
                    node = Mul(node, right, "explicit")
                    flag = rflag
                else:
                    node = Div(node, right)
                    flag = "div" if rflag is None else rflag
                continue
            if tok.kind in _IMPLICIT_START:
                if flag == "div":
                    raise _Abstain(Reason.AMBIGUOUS_INPUT, "implicit product after a division (e.g. 1/2x)")
                if flag == "caret":
                    raise _Abstain(Reason.AMBIGUOUS_INPUT, "implicit product after a ^ power (e.g. x^2y)")
                if flag == "sqrt":
                    raise _Abstain(Reason.AMBIGUOUS_INPUT, "implicit product after an unbracketed root (e.g. √2x)")
                right, rflag = self.parse_power()
                node = Mul(node, right, "implicit")
                flag = rflag
                continue
            if tok.kind == "NUM":
                raise _Abstain(Reason.AMBIGUOUS_INPUT, "number directly after a value (e.g. x2, 2 1/3, 17 900)")
            break
        return node

    def parse_unary(self) -> tuple[Node, Optional[str]]:
        tok = self.peek()
        if tok is not None and tok.kind == "OP" and tok.text in "+-":
            self.take()
            self.enter()
            inner, flag = self.parse_unary()
            self.leave()
            return (Neg(inner) if tok.text == "-" else Pos(inner)), flag
        return self.parse_power()

    def parse_power(self) -> tuple[Node, Optional[str]]:
        base, flag = self.parse_atom()
        tok = self.peek()
        if tok is not None and tok.kind == "SUP":
            self.take()
            node = Pow(base, Number("2" if tok.text == "²" else "3"), "super")
            self._no_stack()
            return node, None
        if tok is not None and tok.kind == "OP" and tok.text == "^":
            self.take()
            exp = self.parse_exponent()
            node = Pow(base, exp, "caret")
            self._no_stack()
            return node, "caret"
        return base, flag

    def _no_stack(self) -> None:
        tok = self.peek()
        if tok is not None and (tok.kind == "SUP" or (tok.kind == "OP" and tok.text == "^")):
            raise _Abstain(Reason.TOO_COMPLEX, "stacked powers")

    def parse_exponent(self) -> Node:
        tok = self.peek()
        sign = None
        if tok is not None and tok.kind == "OP" and tok.text in "+-":
            sign = self.take().text
        node, _ = self.parse_atom()
        _check_exponent(node)
        if sign == "-":
            node = Neg(node)
        elif sign == "+":
            node = Pos(node)
        return node

    def parse_atom(self) -> tuple[Node, Optional[str]]:
        tok = self.peek()
        if tok is None:
            raise _Abstain(Reason.UNPARSEABLE, "unexpected end")
        if tok.kind == "NUM":
            self.take()
            return Number(tok.text), None
        if tok.kind == "VAR":
            self.take()
            return Variable(tok.text), None
        if tok.kind == "PI":
            self.take()
            return Pi(), None
        if tok.kind == "LP":
            self.take()
            self.enter()
            if self.peek() is not None and self.peek().kind == "RP":
                raise _Abstain(Reason.UNPARSEABLE, "empty brackets")
            inner = self.parse_expr()
            if self.peek() is not None and self.peek().kind == "SEP" and self.peek().text == ",":
                raise _Abstain(Reason.UNPARSEABLE, "tuple inside an expression")
            self.expect("RP")
            self.leave()
            return Group(inner), None
        if tok.kind == "SQRT":
            self.take()
            self.enter()
            nxt = self.peek()
            if nxt is not None and nxt.kind == "LP":
                arg, _ = self.parse_atom()
                self.leave()
                inner = arg.inner if isinstance(arg, Group) else arg
                return Sqrt(inner, True), None
            if nxt is not None and nxt.kind in ("NUM", "VAR", "PI"):
                arg, _ = self.parse_atom()
                self.leave()
                return Sqrt(arg, False), "sqrt"
            raise _Abstain(Reason.UNPARSEABLE, "root needs an argument")
        raise _Abstain(Reason.UNPARSEABLE, f"unexpected {tok.text!r}")


def _walk(node):
    yield node
    for v in getattr(node, "__dict__", {}).values():
        if isinstance(v, tuple):
            for x in v:
                yield from _walk(x)
        elif hasattr(v, "__dataclass_fields__"):
            yield from _walk(v)


def _check_exponent(node: Node) -> None:
    for sub in _walk(node):
        if isinstance(sub, Pow):
            raise _Abstain(Reason.TOO_COMPLEX, "power inside an exponent")
        if isinstance(sub, Number) and abs(sub.value) > MAX_EXPONENT:
            raise _Abstain(Reason.TOO_COMPLEX, "exponent too large")


def parse(text, variables=()) -> ParseResult:
    """Parse a typed answer. Never raises."""
    try:
        if not isinstance(text, str):
            return ParseResult(verdict=abstain(Reason.UNPARSEABLE, "not a string"))
        if len(text) > MAX_CHARS:
            return ParseResult(verdict=abstain(Reason.TOO_LONG, "answer too long"))
        s = _normalise(text)
        if not s:
            return ParseResult(verdict=abstain(Reason.EMPTY_INPUT, "empty answer"))
        toks = _tokenize(s, frozenset(variables))
        node = _Parser(toks).parse_answer()
        return ParseResult(node=node)
    except _Abstain as e:
        return ParseResult(verdict=abstain(e.reason, e.detail))
    except RecursionError:
        return ParseResult(verdict=abstain(Reason.TOO_COMPLEX, "too deep"))
    except Exception as e:  # internal-error guard: never expected
        return ParseResult(verdict=abstain(Reason.WORKER_ERROR, f"internal parser error: {type(e).__name__}"))


def walk(node):
    """Yield every node in the tree (pre-order)."""
    yield from _walk(node)
