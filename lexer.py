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

@dataclass
class OpenParenthesis(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\(")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class CloseParenthesis(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\)")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class OpenBrace(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"{")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class CloseBrace(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"}")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class Semicolon(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r";")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class Comment(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\/\/.*|\/\*[\s\S]*?\*\/")

    @classmethod
    def from_match(cls, m):
        return cls()
    
@dataclass
class CompilerDirective(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"^#.*", re.MULTILINE)

    @classmethod
    def from_match(cls, m):
        return cls()
    
@dataclass
class Tilde(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"~")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class Hyphen(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"-")

    @classmethod
    def from_match(cls, m):
        return cls()
    
@dataclass
class PlusSign(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\+")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class Asterisk(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\*")

    @classmethod
    def from_match(cls, m):
        return cls()       


@dataclass
class ForwardSlash(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"/")

    @classmethod
    def from_match(cls, m):
        return cls()


@dataclass
class PercentSign(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"%")

    @classmethod
    def from_match(cls, m):
        return cls()

    
@dataclass
class Decrement(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"--")

    @classmethod
    def from_match(cls, m):
        raise SyntaxError("Unrecognized token: --")
    

@dataclass
class Ampersand(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"&")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class Caret(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\^")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class Pipe(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"\|")

    @classmethod
    def from_match(cls, m):
        return cls()
    
@dataclass
class LeftShift(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r"<<")

    @classmethod
    def from_match(cls, m):
        return cls()

@dataclass
class RightShift(Token):
    pattern: ClassVar[Pattern[str]] = re.compile(r">>")

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