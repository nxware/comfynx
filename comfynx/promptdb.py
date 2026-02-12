
import json
import glob
import re
from typing import List, Optional, Tuple, Any
import os.path
import urllib.parse
import base64
from io import BytesIO
from PIL import Image

from nwebclient import runner as r
from nwebclient import util as u
from nwebclient import base as b
from nwebclient import web as w
from nwebclient import dev as d

# --- Helper type ---
Pos = Tuple[int, int, int, int, int, int]  
# (start_index, end_index, start_line, start_col, end_line, end_col)


class Token:
    __match_args__ = ("value", "start", "end")
    def __init__(self, value: str, start: int = 0, end: int = None, text: str = None, owner=None):
        self.value = value
        self.start = start
        self.end = end
        self.owner = owner
        if text is None:
            text = value
            end = len(value)
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

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r}, pos={self.pos()})"
    
    @property
    def html(self):
        return w.span(self.value)
    
    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if not isinstance(other, Token):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.value == other.value
            #and self.start == other.start
            #and self.end == other.end
        )

    def __hash__(self) -> int:
        return hash((type(self), self.value)) # self.start, self.end

    def __ne__(self, other: Any) -> bool:
        return not self == other


class WhiteSpaceToken(Token):
    pass

class MarkToken(Token):
    """ e.g. . , : """
    pass

class WordToken(Token):
    @property
    def html(self):
        attrs = dict(title='')
        if self.owner is not None and self.owner.owner is not None:
            attrs['title'] += self.owner.owner.get_word_translation(self.value) + ' '
            classes = self.owner.owner.get_word_classes(self.value)
            if len(classes) > 0 and 'neutral' not in classes:
                attrs['class'] = classes[0]
                attrs['title'] += classes[0] + ' '
        return w.span(self.value, **attrs)

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
    def __init__(self, open_char: str, start: int, end: int, text: str, owner):
        super().__init__(open_char, start, end, text, owner)
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
    
    def __init__(self, items=None,owner=None):
        super().__init__()
        if items is not None:
            for itm in items:
                self.append(itm)
        self.owner = owner

    def words(self):
        return [t.value for t in self if isinstance(t, WordToken)]

    def distinct_words(self):
        res = []
        for w in [t for t in self if isinstance(t, WordToken)]:
            if w.value not in res:
                res.append(w.value)
        return res
    
    def word_bigrams(self, sentence_marks=[".", "!", "?"]):
        """
        Liefert Bigramme aus WordToken-Paaren.
        Satzgrenzen unterbrechen die Bigrammbildung.
        """
        bigrams = []
        prev_word = None

        for token in self:
            # Satzgrenze → Reset
            if isinstance(token, MarkToken) and token.value in sentence_marks:
                prev_word = None
                continue

            if isinstance(token, WordToken):
                if prev_word is not None:
                    bigrams.append((prev_word, token))
                prev_word = token
            else:
                # alles andere ignorieren (Whitespace etc.)
                continue

        return bigrams
    
    def word_ngrams(self, n: int, sentence_marks={".", "!", "?"}) -> list[Tuple[WordToken, ...]]:
        """
        Liefert n-Gramme aus WordToken-Tupeln.
        Satzgrenzen unterbrechen die n-Gramm-Bildung.

        n=1 → Unigramme
        n=2 → Bigramme
        n=3 → Trigramme
        """
        if n <= 0:
            raise ValueError("n muss >= 1 sein")
        ngrams = []
        window = []
        for token in self:

            if isinstance(token, MarkToken) and token.value in sentence_marks:
                window.clear()
                continue
            if isinstance(token, WordToken):
                window.append(token)
                if len(window) == n:
                    ngrams.append(tuple(window))
                    window.pop(0) # Sliding window
        return ngrams
        


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
            tokens.append(WhiteSpaceToken(value, start, end, text, tokens))
        elif kind == "LITERAL":
            tokens.append(LiteralToken(value, start, end, text, tokens))
        elif kind in ("LPAREN", "RPAREN"):
            # Wir nutzen weiterhin Token-Objekte hier, Parser macht daraus BracketToken
            tokens.append(Token(value, start, end, text, tokens))
        elif kind == "MARK":
            tokens.append(MarkToken(value, start, end, text, tokens))
        elif kind == "WORD":
            tokens.append(WordToken(value, start, end, text, tokens))
        else:  # OTHER
            tokens.append(Token(value, start, end, text, tokens))

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


class PromptDB(r.BaseJobExecutor):
    def __init__(self, path=None, args: u.Args = None):
        super().__init__('promptdb')
        if args is None:
            args = u.Args()
        self.args = args
        self.prompts = []
        self.cats = {}
        self.words = {}
        self.path = None
        self.define_sig(d.PStr('op', 'scandir'))
        self.define_sig(d.PStr('op', 'save'))
        if path is not None:
            self.path = path
            with open(os.path.join(path, 'prompts.json'), 'r') as f:
                j = json.load(f)
                self.prompts = j.get('prompts', [])
            with open(os.path.join(path, 'cats.json'), 'r') as f:
                self.cats  = json.load(f)
            if os.path.isfile(os.path.join(path, 'words.json')):
                with open(os.path.join(path, 'words.json'), 'r') as f:
                    self.words  = json.load(f)

    def save(self, path):
        with open(os.path.join(path, 'prompts.json'), 'w') as f:
            json.dump({
                'prompts': self.prompts
            }, f)
        with open(os.path.join(path, 'cats.json'), 'w') as f:
            json.dump(self.cats, f)

    def execute_save(self, data={}):
        if 'prompt' in data:
            self.prompts.append(data['prompt'])
        self.save(self.path)
        return self.success()

    def execute_class_add_word(self, data):
        cls = data['cls']
        word = data['word']
        self.cats[cls].append(word)
        return self.success(ui=dict(toast='Added'))
    
    def execute_class_remove_word(self, data):
        cls = data['cls']
        word = data['word']
        self.cats[cls].remove(word)
        return self.success(ui=dict(toast='Removed'))

    def get_word_translation(self, value):
        return self.words.get(value, '')

    def get_all_classes(self):
        return self.cats.keys()

    def get_word_class(self, value):
        for k, v in self.cats.items():
            if value in v:
                return k
        return None
    
    def get_word_classes(self, value):
        res = []
        for k, v in self.cats.items():
            if value in v:
                res.append(k)
        return res

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
        self.info(f"Scan-Dir: {path}")
        i = 0
        for path in glob.glob(os.path.join(path, '**/**.png'), recursive=recursive):
            self.info(f"Processing: {path}")
            from PIL import Image
            im = Image.open(path)
            p = self.extract_prompt(im)
            self.add_prompt(p)
            i += 1
        return i

    def execute_scandir(self, data={}):
        path = data.get('path', self.args.get('comfyui', {}).get('output_path', '.'))
        return self.success(count=self.scan_dir(path=path, recursive=True))

    def tokenize(self, prompt) -> Tokens:
        res = tokenize(prompt)
        res.owner = self
        return res

    def setupPage(self, p: b.Page):
        p.style("""
            span.person { color: #aa0000; }
            span.environment { color: #00aa00; }
            span.composition { color: #aa00aa; }
            span.photo { color: #aa00aa; }
        """)

    def part_prompt(self, p: b.Page, params={}):
        prompt = params.get('prompt')
        p(f'<input type="hidden" name="prompt" id="prompt" value="'+urllib.parse.quote(prompt)+'" />')
        for t in self.tokenize(prompt):
            if isinstance(t, WordToken):
                p(w.a(t.html, self.link_word(t.value)))
            else:
                p(t.html)
        p.hr()
        p(self.action_btn_parametric("Send to Comfy", {
            "type": "comfyui",
            "prompt": "#prompt",
            "op": "queue"
        }))
        p(self.action_btn_parametric("Save", {'type': self.type, "prompt": "#prompt", 'op': 'save'}))

    def execute_image(self, data):
        with BytesIO(base64.b64decode(data['file_data'])) as image_stream:
            prompt = self.extract_prompt(Image.open(image_stream))
            plink = f'?type={self.type}&{self.type}=prompt&prompt=' +  urllib.parse.quote(prompt)
            return self.success(ui=dict(prepend=w.a("Prompt", plink)))

    def get_top_ngrams(self, word, n=3, top=25):
        stats = u.CountList()
        for pstr in self.prompts:
            pt = self.tokenize(pstr)
            stats.add_all(pt.word_ngrams(3, sentence_marks=['.',';',',']))
        def keep_fn(tokens):
            for t in tokens:
                if word == t.value:
                    return True
            return False
        stats = u.CountList(*list(filter(keep_fn, stats)))
        return stats.top_n(top)

    def part_word(self, p: b.Page, params={}):
        word = params['word']
        ts = Tokens(owner=self)
        wt = WordToken(word, owner=ts)
        p(wt.html)
        for c in self.cats.keys():
            is_in = word in self.cats[c]
            p(w.checkbox_npy(c, is_in, dict(type=self.type, op='class_add_word', word=word, cls=c), dict(type=self.type, op='class_remove_word', word=word, cls=c)))
        p.ul(map(self.tag_words, self.get_top_ngrams(word)))

    def part_words(self, p: b.Page, params={}):
        words = u.read_array_from_dict(params, 'w')
        p(str(words))

    def link_word(self, word):
        return self.link(self.part_word, 'word=' + urllib.parse.quote(word))

    def link_words(self, words):
        qd = {}
        i = 1
        for word in words:
            qd[f'w{i}'] = str(word)
            i += 1
        q = w.ql(qd)
        return self.link(self.part_words, q[1:])
    
    def tag_word(self, word):
        return w.a(word, self.link_word(word))
    
    def tag_words(self, words):
        title = ' '.join(map(lambda x: x.value, words))
        return w.a(title, self.link_words(words))

    def part_stats(self, p: b.Page, params={}):
        stats = u.CountList()
        bi_stats = u.CountList()
        ti_stats = u.CountList()
        for pstr in self.prompts:
            pt = self.tokenize(str(pstr))
            stats.add_all(pt.words())
            bi_stats.add_all(pt.word_bigrams(sentence_marks=['.',';',',']))
            ti_stats.add_all(pt.word_ngrams(3, sentence_marks=['.',';',',']))
        p.ul( [f'{self.tag_word(k)}: {v}' for k, v in stats.top_n(50).items()])
        p.ul( [f'{self.tag_words(k)}: {v}' for k, v in bi_stats.top_n(50).items()])
        p.ul( [f'{self.tag_words(k)}: {v}' for k, v in ti_stats.top_n(50).items()]) # TODO link

    def part_index(self, p: b.Page, params={}):
        p.h1("PromptDB")
        with p.t('form', action='?') as f:
            p(w.hidden("type",self.type))
            p(w.hidden(self.type, "prompt"))
            p(w.textarea('', name='prompt'))
            p.input('submit', type='submit',value="Analyse")
        p(w.hidden("image_data", '', id="image_data"))
        p.js_ready('nx_initFileDragArea("dropZone", "image_data");')
        p(w.dropzone(title="Bild hier ablegen"))
        p(self.action_btn_parametric("Analyse", dict(type= self.type, op= 'image', file_data= '#image_data')))
        p.ul([
            w.a("Stats", self.link(self.part_stats))
        ])
        p.pre('', id="result")
        p.prop("Prompts", len(self.prompts))


async def nx_aio_request(request):
    headers = dict(request.headers)
    query_params = dict(request.query)
    body = {}
    if request.content_type == 'application/json':
        body = await request.json()
    elif request.content_type == 'application/x-www-form-urlencoded':
        body = dict(await request.post())
    elif request.content_type.startswith('multipart/'):
        reader = await request.multipart
        async for part in reader:
            # Name des Multipart-Feldes
            name = part.name
            # Gesamten Inhalt als bytes lesen
            data = await part.read()
            body[name] = {
                "content_type": part.headers.get("Content-Type"),
                "data": data,              # bytes
                "size": len(data),
            }
    else:
        body = { 'value': await request.text() }
    return {"path": request.path, **headers, **query_params, **body}


async def nx_aio_response(response):
    from aiohttp import web
    return web.Response(text=response)


def init():
    try:
        from server import PromptServer
        from aiohttp import web
        routes = PromptServer.instance.routes
        @routes.get('/nx/promptdb')
        async def my_function(request):
            param1 = request.rel_url.query['name'] # MultiDictProxy
            # request.post()
            text = "Test result"
            return nx_aio_response(text)
    except Exception as e:
        print("Error: PromptDB init " + str(e))