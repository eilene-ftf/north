"""Lexical analysis for the FORTH programming language."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

__all__ = ["WordType", "Word", "lex", "LexicalAnalysisError"]


class WordType(Enum):
    """Tag denoting what kind of
    `Core word <https://forth-standard.org/standard/core#wordlist:core>`_
    that the ``Word`` is.

    **Word types**:

    The follow is a list of all possible words that the lexical analysis
    will recognize. The format is:

    > source level representation: `WordType`

    + `!`: `PUT`
    + `#`: `HASHTAG`
    + `#>`: `HASHTAG_GREATER_THAN`
    + `#S`: `HASH_TAG_S`
    + `'`: `SINGLE_QUOTE`
    + `(`: `LEFT_PAREN`
    + `*`: `ASTERISK`
    + `*/`: `ASTERISK_SLASH`
    + `*/MOD`: `ASTERISK_SLASH_MOD`
    + `+`: `PLUS`
    + `+!`: `PLUS_STORE`
    + `+LOOP`: `PLUS_LOOP`
    + `,`: `COMMA`
    + `-`: `MINUS`
    + `.`: `DOT`
    + `."`: `DOT_QUOTE`
    + `/`: `SLASH`
    + `/MOD`: `SLASH_MOD`
    + `0<`: `ZERO_LESS`
    + `0=`: `ZERO_EQUALS`
    + `1+`: `ONE_PLUS`
    + `1-`: `ONE_MINUS`
    + `2!`: `TWO_STORE`
    + `2*`: `TWO_TIMES`
    + `2/`: `TWO_DIVIDE`
    + `2@`: `TWO_PEEP`
    + `2DROP`: `TWO_DROP`
    + `2DUP`: `TWO_DUP`
    + `2OVER`: `TWO_OVER`
    + `2SWAP`: `TWO_SWAP`
    + `:`: `FUNC`
    + `;`: `END`
    + `<`: `LESS_THAN`
    + `<#`: `LESS_THAN_HASH`
    + `=`: `EQUALS`
    + `>`: `GREATER_THAN`
    + `>BODY`: `TO_BODY`
    + `>IN`: `TO_IN`
    + `>NUMBER`: `TO_NUMBER`
    + `>R`: `PUSHRET`
    + `?DUP`: `QUESTION_DUP`
    + `@`: `PEEP`
    + `ABORT`: `ABORT`
    + `ABORT"`: `ABORT_QUOTE`
    + `ABS`: `ABS`
    + `ACCEPT`: `ACCEPT`
    + `ALIGN`: `ALIGN`
    + `ALIGNED`: `ALIGNED`
    + `ALLOT`: `ALLOT`
    + `AND`: `AND`
    + `BASE`: `BASE`
    + `BEGIN`: `BEGIN`
    + `BL`: `BL`
    + `C!`: `C_STORE`
    + `C,`: `C_COMMA`
    + `C@`: `C_PEEP`
    + `CELL+`: `CELL_PLUS`
    + `CELLS`: `CELLS`
    + `CHAR`: `CHAR`
    + `CHAR+`: `CHAR_PLUS`
    + `CHARS`: `CHARS`
    + `CONSTANT`: `CONSTANT`
    + `COUNT`: `COUNT`
    + `CR`: `CR`
    + `CREATE`: `CREATE`
    + `DECIMAL`: `DECIMAL`
    + `DEPTH`: `DEPTH`
    + `DO`: `DO`
    + `DOES>`: `DOES`
    + `DROP`: `DROP`
    + `DUP`: `DUP`
    + `ELSE`: `ELSE`
    + `EMIT`: `EMIT`
    + `ENVIRONMENT?`: `ENVIRONMENT_QUERY`
    + `EVALUATE`: `EVALUATE`
    + `EXECUTE`: `EXECUTE`
    + `EXIT`: `EXIT`
    + `FILL`: `FILL`
    + `FIND`: `FIND`
    + `FM/MOD`: `FM_SLASH_MOD`
    + `HERE`: `HERE`
    + `HOLD`: `HOLD`
    + `I`: `I`
    + `IF`: `IF`
    + `IMMEDIATE`: `IMMEDIATE`
    + `INVERT`: `INVERT`
    + `J`: `J`
    + `KEY`: `KEY`
    + `LEAVE`: `LEAVE`
    + `LITERAL`: `LITERAL`
    + `LOOP`: `LOOP`
    + `LSHIFT`: `LSHIFT`
    + `M*`: `M_TIMES`
    + `MAX`: `MAX`
    + `MIN`: `MIN`
    + `MOD`: `MOD`
    + `MOVE`: `MOVE`
    + `NEGATE`: `NEGATE`
    + `OR`: `OR`
    + `OVER`: `OVER`
    + `POSTPONE`: `POSTPONE`
    + `QUIT`: `QUIT`
    + `R>`: `POPRET`
    + `R@`: `R_PEEP`
    + `RECURSE`: `RECURSE`
    + `REPEAT`: `REPEAT`
    + `ROT`: `ROT`
    + `RSHIFT`: `RSHIFT`
    + `S"`: `S_QUOTE`
    + `S>D`: `S_TO_D`
    + `SIGN`: `SIGN`
    + `SM/REM`: `SM_SLASH_REM`
    + `SOURCE`: `SOURCE`
    + `SPACE`: `SPACE`
    + `SPACES`: `SPACES`
    + `STATE`: `STATE`
    + `SWAP`: `SWAP`
    + `THEN`: `THEN`
    + `TYPE`: `TYPE`
    + `U.`: `U_DOT`
    + `U<`: `U_LESS`
    + `UM*`: `UM_TIMES`
    + `UM/MOD`: `UM_SLASH_MOD`
    + `UNLOOP`: `UNLOOP`
    + `UNTIL`: `UNTIL`
    + `VARIABLE`: `VARIABLE`
    + `WHILE`: `WHILE`
    + `WORD`: `WORD`
    + `XOR`: `XOR`
    + `[`: `LEFT_BRACKET`
    + `[']`: `BRACKET_TICK`
    + `[CHAR]`: `BRACKET_CHAR`
    + `]`: `RIGHT_BRACKET`

    Identifiers are alphanumeric strings.
    A special marker `EOF` is used to denote the end of the file.
    """

    PUT = auto()
    HASHTAG = auto()
    HASHTAG_GREATER_THAN = auto()
    HASH_TAG_S = auto()
    SINGLE_QUOTE = auto()
    LEFT_PAREN = auto()
    ASTERISK = auto()
    ASTERISK_SLASH = auto()
    ASTERISK_SLASH_MOD = auto()
    PLUS = auto()
    PLUS_STORE = auto()
    PLUS_LOOP = auto()
    COMMA = auto()
    MINUS = auto()
    DOT = auto()
    DOT_QUOTE = auto()
    SLASH = auto()
    SLASH_MOD = auto()
    ZERO_LESS = auto()
    ZERO_EQUALS = auto()
    ONE_PLUS = auto()
    ONE_MINUS = auto()
    TWO_STORE = auto()
    TWO_TIMES = auto()
    TWO_DIVIDE = auto()
    TWO_PEEP = auto()
    TWO_DROP = auto()
    TWO_DUP = auto()
    TWO_OVER = auto()
    TWO_SWAP = auto()
    FUNC = auto()
    END = auto()
    LESS_THAN = auto()
    LESS_THAN_HASH = auto()
    EQUALS = auto()
    GREATER_THAN = auto()
    TO_BODY = auto()
    TO_IN = auto()
    TO_NUMBER = auto()
    PUSHRET = auto()
    QUESTION_DUP = auto()
    PEEP = auto()
    ABORT = auto()
    ABORT_QUOTE = auto()
    ABS = auto()
    ACCEPT = auto()
    ALIGN = auto()
    ALIGNED = auto()
    ALLOT = auto()
    AND = auto()
    BASE = auto()
    BEGIN = auto()
    BL = auto()
    C_STORE = auto()
    C_COMMA = auto()
    C_PEEP = auto()
    CELL_PLUS = auto()
    CELLS = auto()
    CHAR = auto()
    CHAR_PLUS = auto()
    CHARS = auto()
    CONSTANT = auto()
    COUNT = auto()
    CR = auto()
    CREATE = auto()
    DECIMAL = auto()
    DEPTH = auto()
    DO = auto()
    DOES = auto()
    DROP = auto()
    DUP = auto()
    ELSE = auto()
    EMIT = auto()
    ENVIRONMENT_QUERY = auto()
    EVALUATE = auto()
    EXECUTE = auto()
    EXIT = auto()
    FILL = auto()
    FIND = auto()
    FM_SLASH_MOD = auto()
    HERE = auto()
    HOLD = auto()
    I = auto()
    IF = auto()
    IMMEDIATE = auto()
    INVERT = auto()
    J = auto()
    KEY = auto()
    LEAVE = auto()
    LITERAL = auto()
    LOOP = auto()
    LSHIFT = auto()
    M_TIMES = auto()
    MAX = auto()
    MIN = auto()
    MOD = auto()
    MOVE = auto()
    NEGATE = auto()
    OR = auto()
    OVER = auto()
    POSTPONE = auto()
    QUIT = auto()
    POPRET = auto()
    R_PEEP = auto()
    RECURSE = auto()
    REPEAT = auto()
    ROT = auto()
    RSHIFT = auto()
    S_QUOTE = auto()
    S_TO_D = auto()
    SIGN = auto()
    SM_SLASH_REM = auto()
    SOURCE = auto()
    SPACE = auto()
    SPACES = auto()
    STATE = auto()
    SWAP = auto()
    THEN = auto()
    TYPE = auto()
    U_DOT = auto()
    U_LESS = auto()
    UM_TIMES = auto()
    UM_SLASH_MOD = auto()
    UNLOOP = auto()
    UNTIL = auto()
    VARIABLE = auto()
    WHILE = auto()
    WORD = auto()
    XOR = auto()
    LEFT_BRACKET = auto()
    BRACKET_TICK = auto()
    BRACKET_CHAR = auto()
    RIGHT_BRACKET = auto()
    IDENT = auto()
    EOF = auto()


@dataclass
class Word:
    """Tagged union for as a source-level representation of user FORTH code.

    Attributes:
        tag (WordType): The tag denoting what kind of word the `Word` is.
        cont (str): The raw string contents.

    Examples:
    ```
    word = Word(VARIABLE, "foo", range(len("foo")))
    ```
    """

    tag: WordType
    cont: str


WORD_TAG_DICT: dict[str, WordType] = {
    "!": WordType.PUT,
    "#": WordType.HASHTAG,
    "#>": WordType.HASHTAG_GREATER_THAN,
    "#S": WordType.HASH_TAG_S,
    "'": WordType.SINGLE_QUOTE,
    "(": WordType.LEFT_PAREN,
    "*": WordType.ASTERISK,
    "*/": WordType.ASTERISK_SLASH,
    "*/MOD": WordType.ASTERISK_SLASH_MOD,
    "+": WordType.PLUS,
    "+!": WordType.PLUS_STORE,
    "+LOOP": WordType.PLUS_LOOP,
    ",": WordType.COMMA,
    "-": WordType.MINUS,
    ".": WordType.DOT,
    '."': WordType.DOT_QUOTE,
    "/": WordType.SLASH,
    "/MOD": WordType.SLASH_MOD,
    "0<": WordType.ZERO_LESS,
    "0=": WordType.ZERO_EQUALS,
    "1+": WordType.ONE_PLUS,
    "1-": WordType.ONE_MINUS,
    "2!": WordType.TWO_STORE,
    "2*": WordType.TWO_TIMES,
    "2/": WordType.TWO_DIVIDE,
    "2@": WordType.TWO_PEEP,
    "2DROP": WordType.TWO_DROP,
    "2DUP": WordType.TWO_DUP,
    "2OVER": WordType.TWO_OVER,
    "2SWAP": WordType.TWO_SWAP,
    ":": WordType.FUNC,
    ";": WordType.END,
    "<": WordType.LESS_THAN,
    "<#": WordType.LESS_THAN_HASH,
    "=": WordType.EQUALS,
    ">": WordType.GREATER_THAN,
    ">BODY": WordType.TO_BODY,
    ">IN": WordType.TO_IN,
    ">NUMBER": WordType.TO_NUMBER,
    ">R": WordType.PUSHRET,
    "?DUP": WordType.QUESTION_DUP,
    "@": WordType.PEEP,
    "ABORT": WordType.ABORT,
    'ABORT"': WordType.ABORT_QUOTE,
    "ABS": WordType.ABS,
    "ACCEPT": WordType.ACCEPT,
    "ALIGN": WordType.ALIGN,
    "ALIGNED": WordType.ALIGNED,
    "ALLOT": WordType.ALLOT,
    "AND": WordType.AND,
    "BASE": WordType.BASE,
    "BEGIN": WordType.BEGIN,
    "BL": WordType.BL,
    "C!": WordType.C_STORE,
    "C,": WordType.C_COMMA,
    "C@": WordType.C_PEEP,
    "CELL+": WordType.CELL_PLUS,
    "CELLS": WordType.CELLS,
    "CHAR": WordType.CHAR,
    "CHAR+": WordType.CHAR_PLUS,
    "CHARS": WordType.CHARS,
    "CONSTANT": WordType.CONSTANT,
    "COUNT": WordType.COUNT,
    "CR": WordType.CR,
    "CREATE": WordType.CREATE,
    "DECIMAL": WordType.DECIMAL,
    "DEPTH": WordType.DEPTH,
    "DO": WordType.DO,
    "DOES>": WordType.DOES,
    "DROP": WordType.DROP,
    "DUP": WordType.DUP,
    "ELSE": WordType.ELSE,
    "EMIT": WordType.EMIT,
    "ENVIRONMENT?": WordType.ENVIRONMENT_QUERY,
    "EVALUATE": WordType.EVALUATE,
    "EXECUTE": WordType.EXECUTE,
    "EXIT": WordType.EXIT,
    "FILL": WordType.FILL,
    "FIND": WordType.FIND,
    "FM/MOD": WordType.FM_SLASH_MOD,
    "HERE": WordType.HERE,
    "HOLD": WordType.HOLD,
    "I": WordType.I,
    "IF": WordType.IF,
    "IMMEDIATE": WordType.IMMEDIATE,
    "INVERT": WordType.INVERT,
    "J": WordType.J,
    "KEY": WordType.KEY,
    "LEAVE": WordType.LEAVE,
    "LITERAL": WordType.LITERAL,
    "LOOP": WordType.LOOP,
    "LSHIFT": WordType.LSHIFT,
    "M*": WordType.M_TIMES,
    "MAX": WordType.MAX,
    "MIN": WordType.MIN,
    "MOD": WordType.MOD,
    "MOVE": WordType.MOVE,
    "NEGATE": WordType.NEGATE,
    "OR": WordType.OR,
    "OVER": WordType.OVER,
    "POSTPONE": WordType.POSTPONE,
    "QUIT": WordType.QUIT,
    "R>": WordType.POPRET,
    "R@": WordType.R_PEEP,
    "RECURSE": WordType.RECURSE,
    "REPEAT": WordType.REPEAT,
    "ROT": WordType.ROT,
    "RSHIFT": WordType.RSHIFT,
    'S"': WordType.S_QUOTE,
    "S>D": WordType.S_TO_D,
    "SIGN": WordType.SIGN,
    "SM/REM": WordType.SM_SLASH_REM,
    "SOURCE": WordType.SOURCE,
    "SPACE": WordType.SPACE,
    "SPACES": WordType.SPACES,
    "STATE": WordType.STATE,
    "SWAP": WordType.SWAP,
    "THEN": WordType.THEN,
    "TYPE": WordType.TYPE,
    "U.": WordType.U_DOT,
    "U<": WordType.U_LESS,
    "UM*": WordType.UM_TIMES,
    "UM/MOD": WordType.UM_SLASH_MOD,
    "UNLOOP": WordType.UNLOOP,
    "UNTIL": WordType.UNTIL,
    "VARIABLE": WordType.VARIABLE,
    "WHILE": WordType.WHILE,
    "WORD": WordType.WORD,
    "XOR": WordType.XOR,
    "[": WordType.LEFT_BRACKET,
    "[']": WordType.BRACKET_TICK,
    "[CHAR]": WordType.BRACKET_CHAR,
    "]": WordType.RIGHT_BRACKET,
}


def wordtype2str(x: WordType) -> str:
    return str(x).removeprefix("WordType.")


@dataclass
class LexicalAnalysisError(Exception):
    msg: str


def lex(src: str) -> list[Word]:
    """Perform lexical analysis on the string contents of a source-level program.

    Args:
        src (str): The source-level program, either read from a file or from input.

    Returns:
        A list of the lexically analyzed words.

    Raises:
        `LexicalAnalysisError`.
    """

    src_words = src.upper().split()
    word_rec = []
    for word in src_words:
        if word in WORD_TAG_DICT:
            word_rec.append(Word(WORD_TAG_DICT[word], word))
        else:
            word_rec.append(Word(WordType.IDENT, word))
    word_rec.append(Word(WordType.EOF, "<EOF>"))
    return word_rec
