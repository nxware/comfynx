
import json
import glob
import re
from typing import List, Optional, Tuple

# --- Helper type ---
Pos = Tuple[int, int, int, int, int, int]  
# (start_index, end_index, start_line, start_col, end_line, end_col)


class Token:
    def __init__(self, value: str, start: int, end: int, text: str):
        self.value = value
        self.start = start
        self.end = end
        # compute line/col positions
        self.start_line, self.start_col = self._line_col_at(text, start)
        self.end_line, self.end_col = self._line_col_at(text, end - 1)  # end-1 ist letztes Zeichen
    def _line_col_at(self, text: str, index: int) -> Tuple[int, int]:
        # Zeile: 1-basiert, Spalte: 1-basiert
        # index kann -1 wenn end==0 (leer); handle safe
        if index < 0:
            return 1, 1
        line = text.count("\n", 0, index) + 1
        last_n = text.rfind("\n", 0, index)
        col = index - last_n
        return line, col

    def pos(self) -> Pos:
        return (self.start, self.end, self.start_line, self.start_col, self.end_line, self.end_col)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r}, pos={self.pos()})"


class WhiteSpaceToken(Token):
    pass

class MarkToken(Token):
    """ e.g. . , : """
    pass

class WordToken(Token):
    pass

class LiteralToken(Token):
    """ e.g. 'a literal' """
    pass

class BracketToken(Token):
    """
    Repräsentiert einen kompletten Klammerausdruck.
    value = öffnende Klammer (z.B. '(')
    children = Liste enthaltener Token
    close = schließende Klammer (z.B. ')') oder None
    open_pos, close_pos: Pos-Tuples oder None
    """
    def __init__(self, open_char: str, start: int, end: int, text: str):
        super().__init__(open_char, start, end, text)
        self.children: List[Token] = []
        self.close: Optional[str] = None
        self.open_pos: Pos = self.pos()
        self.close_pos: Optional[Pos] = None

    def set_close_pos_from_token(self, tok: Token):
        self.close = tok.value
        self.close_pos = tok.pos()

    def __repr__(self):
        return (f"BracketToken({self.value!r}, open_pos={self.open_pos}, "
                f"close={self.close!r}, close_pos={self.close_pos}, children={self.children})")


class Tokens(list):
    def __init__(self):
        super().__init__()

    def words(self):
        return [t.value for t in self if isinstance(t, WordToken)]

    def distinct_words(self):
        res = []
        for w in [t for t in self if isinstance(t, WordToken)]:
            if w.value not in res:
                res.append(w.value)
        return res
        


# -----------------------
# Tokenizer mit Position
# -----------------------
def tokenize(text: str) -> Tokens:
    """
    Liefert eine flache Liste von Tokens (je Token: value + Positionen).
    Erzeugte Token-Typen: WhiteSpaceToken, LiteralToken, Bracket-Token-Platzhalter (als Token),
    MarkToken, WordToken.
    """
    token_spec = [
        ("WHITESPACE", r"\s+"),
        ("LITERAL",    r"'([^'\\]|\\.)*'"),        # einfache Unterstützung für escaped ' innerhalb
        ("LPAREN",     r"[\(\{\[]"),               # öffnende Klammer
        ("RPAREN",     r"[\)\}\]]"),               # schließende Klammer
        ("MARK",       r"[.,:;!?]"),
        ("WORD",       r"[A-Za-z0-9_]+"),
        ("OTHER",      r".")                       # Fallback für einzelne andere Zeichen
    ]

    pattern = "|".join(f"(?P<{n}>{r})" for n, r in token_spec)
    regex = re.compile(pattern)

    tokens: List[Token] = Tokens()

    for m in regex.finditer(text):
        kind = m.lastgroup
        value = m.group()
        start = m.start()
        end = m.end()

        if kind == "WHITESPACE":
            tokens.append(WhiteSpaceToken(value, start, end, text))
        elif kind == "LITERAL":
            tokens.append(LiteralToken(value, start, end, text))
        elif kind in ("LPAREN", "RPAREN"):
            # Wir nutzen weiterhin Token-Objekte hier, Parser macht daraus BracketToken
            tokens.append(Token(value, start, end, text))
        elif kind == "MARK":
            tokens.append(MarkToken(value, start, end, text))
        elif kind == "WORD":
            tokens.append(WordToken(value, start, end, text))
        else:  # OTHER
            tokens.append(Token(value, start, end, text))

    return tokens

# -----------------------
# Parser für Verschachtelung
# -----------------------
PAIRS = {"(": ")", "{": "}", "[": "]"}
OPEN_SET = set(PAIRS.keys())
CLOSE_SET = set(PAIRS.values())

def parse_brackets(tokens: List[Token]) -> List[Token]:
    """
    Baut aus der flachen Token-Liste verschachtelte BracketToken-Objekte.
    Rückgabe: Liste von Top-Level-Tokens (WordToken, MarkToken, BracketToken, ...).
    """
    stack: List[List[Token]] = []   # Stack von children-Listen
    bracket_stack: List[BracketToken] = []  # zugehörige BracketToken-Objekte
    root: List[Token] = []
    current = root

    for tok in tokens:
        # Für generische Token-Instanzen prüfen wir auf value ob Klammerzeichen
        val = getattr(tok, "value", None)
        if val in OPEN_SET:
            # Erzeuge BracketToken mit Position der offenen Klammer
            bt = BracketToken(val, tok.start, tok.end, reconstruct_text_from_tokens(tokens, tok))
            # WICHTIG: BracketToken braucht korrekten text-Parameter für Zeile/Spalte; 
            # wir übergeben den gesamten Originaltext weiter unten, aber hier nutzen wir Hack:
            # stattdessen: wir wollen Zeile/Spalte korrekt berechnen -> wir brauchen originalen Text.
            # Um das sauber zu halten, der Token hatte bereits positionsberechnung mit dem originaltext im Tokenizer,
            # daher verwenden wir die Positionen aus tok direkt:
            bt = BracketToken(val, tok.start, tok.end, reconstruct_original_text_for_token(tok, tokens))
            # (Wir reparieren das weiter unten in vereinfachter Form.)
            current.append(bt)
            # push current container & switch to bt.children
            stack.append(current)
            bracket_stack.append(bt)
            current = bt.children
        elif val in CLOSE_SET:
            if not stack:
                raise ValueError(f"Unmatched closing bracket at pos {tok.pos()}: {val!r}")
            last_bracket = bracket_stack[-1]
            expected = PAIRS[last_bracket.value]
            if expected != val:
                raise ValueError(f"Mismatched bracket: expected {expected!r} but got {val!r} at {tok.pos()}")
            # set close info on last_bracket
            last_bracket.set_close_pos_from_token(tok)
            # pop back
            bracket_stack.pop()
            current = stack.pop()
        else:
            current.append(tok)

    if stack:
        # offene Klammern übrig
        unclosed = bracket_stack[-1]
        raise ValueError(f"Unclosed bracket starting at {unclosed.open_pos}")

    return root


# Helper functions to compute correct text context for BracketToken constructor.
# (Die vorherige Konstruktion brach die Übergabe des Originaltexts; wir berechnen die Zeile/Spalte
#  direkt aus den Indizes statt über den Konstruktor-Hack.)

def reconstruct_text_for_position(text: str) -> str:
    # einfach Originaltext zurückgeben; viele Tokens haben Konstruktor die text braucht
    return text

def reconstruct_original_text_for_token(tok: Token, tokens_all: List[Token]) -> str:
    # Wir speichern und erwarten, dass der Token in tokenizer bereits mit originalem Text erzeugt wurde,
    # sodass tok.start/tok.end mit diesem Text stimmen. Hier geben wir das Original nicht neu ein,
    # um die einfache Implementierung zu halten. In diesem Beispiel verwenden wir eine global sichtbare
    # Referenz nicht — aber wir brauchen sie nicht, weil wir umgehen und BracketToken-Positionsfelder
    # später manuell setzen.
    return ""  # placeholder; wir setzen Positionen manuell weiter unten


# ---- Alternative, saubere Implementierung: Parser erhält auch den originalen Text als Parameter,
# ---- so dass BracketToken korrekte line/col-Berechnung erhält.
def parse_brackets_with_text(tokens: List[Token], text: str) -> List[Token]:
    stack: List[List[Token]] = []
    bracket_stack: List[BracketToken] = []
    root: List[Token] = []
    current = root

    for tok in tokens:
        val = getattr(tok, "value", None)
        if val in OPEN_SET:
            # Erzeuge BracketToken mit originalem Text, damit line/col korrekt berechnet werden
            bt = BracketToken(val, tok.start, tok.end, text)
            current.append(bt)
            stack.append(current)
            bracket_stack.append(bt)
            current = bt.children
        elif val in CLOSE_SET:
            if not stack:
                raise ValueError(f"Unmatched closing bracket at pos {tok.pos()}: {val!r}")
            last_bracket = bracket_stack[-1]
            expected = PAIRS[last_bracket.value]
            if expected != val:
                raise ValueError(f"Mismatched bracket: expected {expected!r} but got {val!r} at {tok.pos()}")
            # set close info on last_bracket (benutze tok und text für genaue Pos)
            last_bracket.set_close_pos_from_token(tok)
            bracket_stack.pop()
            current = stack.pop()
        else:
            current.append(tok)

    if stack:
        unclosed = bracket_stack[-1]
        raise ValueError(f"Unclosed bracket starting at {unclosed.open_pos}")

    return root


class PromptDB:
    def __init__(self, path=None):
        super().__init__()
        self.prompts = []
        if path is not None:
            with open(path, 'r') as f:
                j = json.load(f)
                self.prompts = j.get('prompts', [])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'prompts': self.prompts
            }, f)

    def distinct_words(self):
        res = []
        for p in self.prompts:
            for w in tokenize(p).distinct_words():
                if w not in res:
                    res.append(w)
        return res

    def add_prompt(self, prompt: str):
        if prompt is not None and prompt != '':
            if prompt not in self.prompts:
                self.prompts.append(prompt)

    @staticmethod
    def extract_prompt(prompt_data):
        try:
            from PIL import Image
            if isinstance(prompt_data, Image.Image):
                prompt_data = json.loads(prompt_data.info['prompt'])
            for k in prompt_data.keys():
                d = prompt_data[k]
                if d.get('class_type') == 'CLIPTextEncode':
                    return d.get('inputs', {}).get('text', '')
            return ''
        except Exception as e:
            print(f"extract_prompt {e}")

    def scan_dir(self, path, recursive=False):
        for path in glob.glob(os.path.join(path, '**/**.png'), recursive=recursive):
            from PIL import Image
            im = Image.open(path)
            p = self.extract_prompt(im)
            self.add_prompt(p)
