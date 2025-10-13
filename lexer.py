from dataclasses import dataclass
from typing import *
import re


KEYWORDS = set(["int", "void", "return"])

class Token:
    pattern: ClassVar[Pattern[str]]

    @classmethod
    def from_match(cls, m: re.Match[str]) -> "Token":
        raise NotImplementedError

@dataclass
class Identifier(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"[a-zA-Z_]\w*\b")
    name: str
    is_keyword: bool

    @classmethod
    def from_match(cls, m):
        name = m.group(0)
        is_keyword = name in KEYWORDS
        return cls(name, is_keyword) 

@dataclass
class Constant(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"[0-9]+\b")
    value: int

    @classmethod
    def from_match(cls, m):
        return cls(int(m.group(0)))

class OpenParenthesis(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\(")

    @classmethod
    def from_match(cls, m):
        return cls()

class CloseParenthesis(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\)")

    @classmethod
    def from_match(cls, m):
        return cls()

class OpenBrace(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"{")

    @classmethod
    def from_match(cls, m):
        return cls()

class CloseBrace(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"}")

    @classmethod
    def from_match(cls, m):
        return cls()

class Semicolon(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r";")

    @classmethod
    def from_match(cls, m):
        return cls()

class Comment(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\/\/.*|\/\*[\s\S]*?\*\/")

    @classmethod
    def from_match(cls, m):
        return cls()

class CompilerDirective(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"^#.*", re.MULTILINE)

    @classmethod
    def from_match(cls, m):
        return cls()

class Tilde(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"~")

    @classmethod
    def from_match(cls, m):
        return cls()

class Hyphen(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"-")

    @classmethod
    def from_match(cls, m):
        return cls()

class PlusSign(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\+")

    @classmethod
    def from_match(cls, m):
        return cls()

class Asterisk(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\*")

    @classmethod
    def from_match(cls, m):
        return cls()       

class ForwardSlash(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"/")

    @classmethod
    def from_match(cls, m):
        return cls()

class PercentSign(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"%")

    @classmethod
    def from_match(cls, m):
        return cls()

class Decrement(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"--")

    @classmethod
    def from_match(cls, m):
        raise SyntaxError("Unrecognized token: --")

class Ampersand(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"&")

    @classmethod
    def from_match(cls, m):
        return cls()

class Caret(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\^")

    @classmethod
    def from_match(cls, m):
        return cls()

class Pipe(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\|")

    @classmethod
    def from_match(cls, m):
        return cls()

class LeftShift(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"<<")

    @classmethod
    def from_match(cls, m):
        return cls()

class RightShift(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r">>")

    @classmethod
    def from_match(cls, m):
        return cls()

class ExclamationMark(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\!")

    @classmethod
    def from_match(cls, m):
        return cls()
    
class TwoAmbersands(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"&&")

    @classmethod
    def from_match(cls, m):
        return cls()

class TwoPipes(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\|\|")

    @classmethod
    def from_match(cls, m):
        return cls()

class DoubleEqualSign(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"==")

    @classmethod
    def from_match(cls, m):
        return cls()

class ExclamationEqualSign(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"!=")

    @classmethod
    def from_match(cls, m):
        return cls()

class LessThan(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"<")

    @classmethod
    def from_match(cls, m):
        return cls()
    
class GreaterThan(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r">")

    @classmethod
    def from_match(cls, m):
        return cls()
    
class LessOrEqual(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"<=")

    @classmethod
    def from_match(cls, m):
        return cls()
    
class GreaterOrEqual(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r">=")

    @classmethod
    def from_match(cls, m):
        return cls()

TOKENS: List["Token"] = [
    Identifier, 
    Constant, 
    OpenParenthesis, 
    CloseParenthesis, 
    OpenBrace, 
    CloseBrace, 
    Semicolon, 
    Comment, 
    CompilerDirective,
    Tilde, 
    Hyphen, 
    Decrement, 
    PlusSign, 
    Asterisk, 
    ForwardSlash, 
    PercentSign,
    Ampersand,
    Caret,
    Pipe,
    LeftShift,
    RightShift,
    ExclamationMark,
    TwoAmbersands,
    TwoPipes,
    DoubleEqualSign,
    ExclamationEqualSign,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual
]
WHITESPACE = re.compile(r"\s*")

def match_token(text: str, pos: int) -> Tuple[int, "Token"]:
    max_pos = pos
    token = None

    for cls in TOKENS:
        pattern = cls.pattern
        m = pattern.match(text, pos)
        if m is not None and m.end() > max_pos:
            token = cls.from_match(m)
            max_pos = m.end()

    return max_pos, token

def get_tokens(text: str):
    i = 0
    tokens = []
    while i < len(text):
        m = WHITESPACE.match(text, i)
        i = m.end()

        if i >= len(text):
            break

        (new_i, token) = match_token(text, i)
        if token is None:
            raise "Error parsing file"
        
        if not isinstance(token, Comment) and not isinstance(token, CompilerDirective):
            tokens.append(token)
        i = new_i
    return tokens