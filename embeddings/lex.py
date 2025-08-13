"""Lexical analysis for the FORTH programming language."""

from dataclasses import dataclass
from enum import Enum, auto

__all__ = ["WordType", "Word", "lex", "LexicalAnalysisError"]


class WordType(Enum):
    """Tag denoting what kind of
    `Core word <https://forth-standard.org/standard/core#wordlist:core>`_
    that the ``Word`` is.

    **Word types**:

    The follow is a list of all possible words that the lexical analysis
    will recognize. The format is:

    > source level representation: `WordType`

    + `!`:      `F_PUT`
    + `#`:      `F_HASHTAG`
    + `#>`:     `F_HASHTAG_GREATER_THAN`
    + `#S`:     `F_HASH_TAG_S`
    + `'`:      `F_SINGLE_QUOTE`
    + `(`:      `F_LEFT_PAREN`
    + `*`:      `F_ASTERISK`
    + `*/`:     `F_ASTERISK_SLASH`
    + `*/MOD`:  `F_ASTERISK_SLASH_MOD`
    + `+`:      `F_PLUS`
    + `+!`:     `F_PLUS_STORE`
    + `+LOOP`:  `F_PLUS_LOOP`
    + `,`:      `F_COMMA`
    + `-`:      `F_MINUS`
    + `.`:      `F_DOT`
    + `."`:     `F_DOT_QUOTE`
    + `/`:      `F_SLASH`
    + `/MOD`:   `F_SLASH_MOD`
    + `0<`:     `F_ZERO_LESS`
    + `0=`:     `F_ZERO_EQUALS`
    + `1+`:     `F_ONE_PLUS`
    + `1-`:     `F_ONE_MINUS`
    + `2!`:     `F_TWO_STORE`
    + `2*`:     `F_TWO_TIMES`
    + `2/`:     `F_TWO_DIVIDE`
    + `2@`:     `F_TWO_PEEP`
    + `2DROP`:  `F_TWO_DROP`
    + `2DUP`:   `F_TWO_DUP`
    + `2OVER`:  `F_TWO_OVER`
    + `2SWAP`:  `F_TWO_SWAP`
    + `:`:      `F_FUNC`
    + `;`:      `F_END`
    + `<`:      `F_LESS_THAN`
    + `<#`:     `F_LESS_THAN_HASH`
    + `=`:      `F_EQUALS`
    + `>`:      `F_GREATER_THAN`
    + `>BODY`:  `F_TO_BODY`
    + `>IN`:    `F_TO_IN`
    + `>NUMBER`: `F_TO_NUMBER`
    + `>R`:     `F_PUSHRET`
    + `?DUP`:   `F_QUESTION_DUP`
    + `@`:      `F_PEEP`
    + `ABORT`:  `F_ABORT`
    + `ABORT"`: `F_ABORT_QUOTE`
    + `ABS`:    `F_ABS`
    + `ACCEPT`: `F_ACCEPT`
    + `ALIGN`:  `F_ALIGN`
    + `ALIGNED`: `F_ALIGNED`
    + `ALLOT`:  `F_ALLOT`
    + `AND`:    `F_AND`
    + `BASE`:   `F_BASE`
    + `BEGIN`:  `F_BEGIN`
    + `BL`:     `F_BL`
    + `C!`:     `F_C_STORE`
    + `C,`:     `F_C_COMMA`
    + `C@`:     `F_C_PEEP`
    + `CELL+`:  `F_CELL_PLUS`
    + `CELLS`:  `F_CELLS`
    + `CHAR`:   `F_CHAR`
    + `CHAR+`:  `F_CHAR_PLUS`
    + `CHARS`:  `F_CHARS`
    + `CONSTANT`: `F_CONSTANT`
    + `COUNT`:  `F_COUNT`
    + `CR`:     `F_CR`
    + `CREATE`: `F_CREATE`
    + `DECIMAL`: `F_DECIMAL`
    + `DEPTH`:  `F_DEPTH`
    + `DO`:     `F_DO`
    + `DOES>`:  `F_DOES`
    + `DROP`:   `F_DROP`
    + `DUP`:    `F_DUP`
    + `ELSE`:   `F_ELSE`
    + `EMIT`:   `F_EMIT`
    + `ENVIRONMENT?`: `F_ENVIRONMENT_QUERY`
    + `EVALUATE`: `F_EVALUATE`
    + `EXECUTE`: `F_EXECUTE`
    + `EXIT`:   `F_EXIT`
    + `FILL`:   `F_FILL`
    + `FIND`:   `F_FIND`
    + `FM/MOD`: `F_FM_SLASH_MOD`
    + `HERE`:   `F_HERE`
    + `HOLD`:   `F_HOLD`
    + `I`:      `F_I`
    + `IF`:     `F_IF`
    + `IMMEDIATE`: `F_IMMEDIATE`
    + `INVERT`: `F_INVERT`
    + `J`:      `F_J`
    + `KEY`:    `F_KEY`
    + `LEAVE`:  `F_LEAVE`
    + `LITERAL`: `F_LITERAL`
    + `LOOP`:   `F_LOOP`
    + `LSHIFT`: `F_LSHIFT`
    + `M*`:     `F_M_TIMES`
    + `MAX`:    `F_MAX`
    + `MIN`:    `F_MIN`
    + `MOD`:    `F_MOD`
    + `MOVE`:   `F_MOVE`
    + `NEGATE`: `F_NEGATE`
    + `OR`:     `F_OR`
    + `OVER`:   `F_OVER`
    + `POSTPONE`: `F_POSTPONE`
    + `QUIT`:   `F_QUIT`
    + `R>`:     `F_POPRET`
    + `R@`:     `F_R_PEEP`
    + `RECURSE`: `F_RECURSE`
    + `REPEAT`: `F_REPEAT`
    + `ROT`:    `F_ROT`
    + `RSHIFT`: `F_RSHIFT`
    + `S"`:     `F_S_QUOTE`
    + `S>D`:    `F_S_TO_D`
    + `SIGN`:   `F_SIGN`
    + `SM/REM`: `F_SM_SLASH_REM`
    + `SOURCE`: `F_SOURCE`
    + `SPACE`:  `F_SPACE`
    + `SPACES`: `F_SPACES`
    + `STATE`:  `F_STATE`
    + `SWAP`:   `F_SWAP`
    + `THEN`:   `F_THEN`
    + `TYPE`:   `F_TYPE`
    + `U.`:     `F_U_DOT`
    + `U<`:     `F_U_LESS`
    + `UM*`:    `F_UM_TIMES`
    + `UM/MOD`: `F_UM_SLASH_MOD`
    + `UNLOOP`: `F_UNLOOP`
    + `UNTIL`:  `F_UNTIL`
    + `VARIABLE`: `F_VARIABLE`
    + `WHILE`:  `F_WHILE`
    + `WORD`:   `F_WORD`
    + `XOR`:    `F_XOR`
    + `[`:      `F_LEFT_BRACKET`
    + `[']`:    `F_BRACKET_TICK`
    + `[CHAR]`: `F_BRACKET_CHAR`
    + `]`:      `F_RIGHT_BRACKET`

    Identifiers are alphanumeric strings. Numbers are for numeric digits.
    A special marker `EOF` is used to denote the end of the file.
    """

    F_PUT = auto()
    F_HASHTAG = auto()
    F_HASHTAG_GREATER_THAN = auto()
    F_HASH_TAG_S = auto()
    F_SINGLE_QUOTE = auto()
    F_LEFT_PAREN = auto()
    F_ASTERISK = auto()
    F_ASTERISK_SLASH = auto()
    F_ASTERISK_SLASH_MOD = auto()
    F_ADD = auto()
    F_ADD_STORE = auto()
    F_ADD_LOOP = auto()
    F_COMMA = auto()
    F_SUB = auto()
    F_DOT = auto()
    F_DOT_QUOTE = auto()
    F_SLASH = auto()
    F_SLASH_MOD = auto()
    F_ZERO_LESS = auto()
    F_ISZERO = auto()
    F_ONE_PLUS = auto()
    F_ONE_MINUS = auto()
    F_TWO_STORE = auto()
    F_TWO_TIMES = auto()
    F_TWO_DIVIDE = auto()
    F_TWO_PEEP = auto()
    F_TWO_DROP = auto()
    F_TWO_DUP = auto()
    F_TWO_OVER = auto()
    F_TWO_SWAP = auto()
    F_FUNC = auto()
    F_END = auto()
    F_LESS_THAN = auto()
    F_LESS_THAN_HASH = auto()
    F_EQUALS = auto()
    F_GREATER_THAN = auto()
    F_TO_BODY = auto()
    F_TO_IN = auto()
    F_TO_NUMBER = auto()
    F_PUSHRET = auto()
    F_QUESTION_DUP = auto()
    F_PEEP = auto()
    F_ABORT = auto()
    F_ABORT_QUOTE = auto()
    F_ABS = auto()
    F_ACCEPT = auto()
    F_ALIGN = auto()
    F_ALIGNED = auto()
    F_ALLOT = auto()
    F_AND = auto()
    F_BASE = auto()
    F_BEGIN = auto()
    F_BL = auto()
    F_C_STORE = auto()
    F_C_COMMA = auto()
    F_C_PEEP = auto()
    F_CELL_PLUS = auto()
    F_CELLS = auto()
    F_CHAR = auto()
    F_CHAR_PLUS = auto()
    F_CHARS = auto()
    F_CONSTANT = auto()
    F_COUNT = auto()
    F_CR = auto()
    F_CREATE = auto()
    F_DECIMAL = auto()
    F_DEPTH = auto()
    F_DO = auto()
    F_DOES = auto()
    F_DROP = auto()
    F_DUP = auto()
    F_ELSE = auto()
    F_EMIT = auto()
    F_ENVIRONMENT_QUERY = auto()
    F_EVALUATE = auto()
    F_EXECUTE = auto()
    F_EXIT = auto()
    F_FILL = auto()
    F_FIND = auto()
    F_FM_SLASH_MOD = auto()
    F_HERE = auto()
    F_HOLD = auto()
    F_I = auto()
    F_IF = auto()
    F_IMMEDIATE = auto()
    F_INVERT = auto()
    F_J = auto()
    F_KEY = auto()
    F_LEAVE = auto()
    F_LITERAL = auto()
    F_LOOP = auto()
    F_LSHIFT = auto()
    F_M_TIMES = auto()
    F_MAX = auto()
    F_MIN = auto()
    F_MOD = auto()
    F_MOVE = auto()
    F_NEGATE = auto()
    F_OR = auto()
    F_OVER = auto()
    F_POSTPONE = auto()
    F_QUIT = auto()
    F_POPRET = auto()
    F_R_PEEP = auto()
    F_RECURSE = auto()
    F_REPEAT = auto()
    F_ROT = auto()
    F_RSHIFT = auto()
    F_S_QUOTE = auto()
    F_S_TO_D = auto()
    F_SIGN = auto()
    F_SM_SLASH_REM = auto()
    F_SOURCE = auto()
    F_SPACE = auto()
    F_SPACES = auto()
    F_STATE = auto()
    F_SWAP = auto()
    F_THEN = auto()
    F_TYPE = auto()
    F_U_DOT = auto()
    F_U_LESS = auto()
    F_UM_TIMES = auto()
    F_UM_SLASH_MOD = auto()
    F_UNLOOP = auto()
    F_UNTIL = auto()
    F_VARIABLE = auto()
    F_WHILE = auto()
    F_WORD = auto()
    F_XOR = auto()
    F_LEFT_BRACKET = auto()
    F_BRACKET_TICK = auto()
    F_BRACKET_CHAR = auto()
    F_RIGHT_BRACKET = auto()
    IDENT = auto()
    NUMBER = auto()
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
    "!": WordType.F_PUT,
    "#": WordType.F_HASHTAG,
    "#>": WordType.F_HASHTAG_GREATER_THAN,
    "#S": WordType.F_HASH_TAG_S,
    "'": WordType.F_SINGLE_QUOTE,
    "(": WordType.F_LEFT_PAREN,
    "*": WordType.F_ASTERISK,
    "*/": WordType.F_ASTERISK_SLASH,
    "*/MOD": WordType.F_ASTERISK_SLASH_MOD,
    "+": WordType.F_ADD,
    "+!": WordType.F_ADD_STORE,
    "+LOOP": WordType.F_ADD_LOOP,
    ",": WordType.F_COMMA,
    "-": WordType.F_SUB,
    ".": WordType.F_DOT,
    '."': WordType.F_DOT_QUOTE,
    "/": WordType.F_SLASH,
    "/MOD": WordType.F_SLASH_MOD,
    "0<": WordType.F_ZERO_LESS,
    "0=": WordType.F_ISZERO,
    "1+": WordType.F_ONE_PLUS,
    "1-": WordType.F_ONE_MINUS,
    "2!": WordType.F_TWO_STORE,
    "2*": WordType.F_TWO_TIMES,
    "2/": WordType.F_TWO_DIVIDE,
    "2@": WordType.F_TWO_PEEP,
    "2DROP": WordType.F_TWO_DROP,
    "2DUP": WordType.F_TWO_DUP,
    "2OVER": WordType.F_TWO_OVER,
    "2SWAP": WordType.F_TWO_SWAP,
    ":": WordType.F_FUNC,
    ";": WordType.F_END,
    "<": WordType.F_LESS_THAN,
    "<#": WordType.F_LESS_THAN_HASH,
    "=": WordType.F_EQUALS,
    ">": WordType.F_GREATER_THAN,
    ">BODY": WordType.F_TO_BODY,
    ">IN": WordType.F_TO_IN,
    ">NUMBER": WordType.F_TO_NUMBER,
    ">R": WordType.F_PUSHRET,
    "?DUP": WordType.F_QUESTION_DUP,
    "@": WordType.F_PEEP,
    "ABORT": WordType.F_ABORT,
    'ABORT"': WordType.F_ABORT_QUOTE,
    "ABS": WordType.F_ABS,
    "ACCEPT": WordType.F_ACCEPT,
    "ALIGN": WordType.F_ALIGN,
    "ALIGNED": WordType.F_ALIGNED,
    "ALLOT": WordType.F_ALLOT,
    "AND": WordType.F_AND,
    "BASE": WordType.F_BASE,
    "BEGIN": WordType.F_BEGIN,
    "BL": WordType.F_BL,
    "C!": WordType.F_C_STORE,
    "C,": WordType.F_C_COMMA,
    "C@": WordType.F_C_PEEP,
    "CELL+": WordType.F_CELL_PLUS,
    "CELLS": WordType.F_CELLS,
    "CHAR": WordType.F_CHAR,
    "CHAR+": WordType.F_CHAR_PLUS,
    "CHARS": WordType.F_CHARS,
    "CONSTANT": WordType.F_CONSTANT,
    "COUNT": WordType.F_COUNT,
    "CR": WordType.F_CR,
    "CREATE": WordType.F_CREATE,
    "DECIMAL": WordType.F_DECIMAL,
    "DEPTH": WordType.F_DEPTH,
    "DO": WordType.F_DO,
    "DOES>": WordType.F_DOES,
    "DROP": WordType.F_DROP,
    "DUP": WordType.F_DUP,
    "ELSE": WordType.F_ELSE,
    "EMIT": WordType.F_EMIT,
    "ENVIRONMENT?": WordType.F_ENVIRONMENT_QUERY,
    "EVALUATE": WordType.F_EVALUATE,
    "EXECUTE": WordType.F_EXECUTE,
    "EXIT": WordType.F_EXIT,
    "FILL": WordType.F_FILL,
    "FIND": WordType.F_FIND,
    "FM/MOD": WordType.F_FM_SLASH_MOD,
    "HERE": WordType.F_HERE,
    "HOLD": WordType.F_HOLD,
    "I": WordType.F_I,
    "IF": WordType.F_IF,
    "IMMEDIATE": WordType.F_IMMEDIATE,
    "INVERT": WordType.F_INVERT,
    "J": WordType.F_J,
    "KEY": WordType.F_KEY,
    "LEAVE": WordType.F_LEAVE,
    "LITERAL": WordType.F_LITERAL,
    "LOOP": WordType.F_LOOP,
    "LSHIFT": WordType.F_LSHIFT,
    "M*": WordType.F_M_TIMES,
    "MAX": WordType.F_MAX,
    "MIN": WordType.F_MIN,
    "MOD": WordType.F_MOD,
    "MOVE": WordType.F_MOVE,
    "NEGATE": WordType.F_NEGATE,
    "OR": WordType.F_OR,
    "OVER": WordType.F_OVER,
    "POSTPONE": WordType.F_POSTPONE,
    "QUIT": WordType.F_QUIT,
    "R>": WordType.F_POPRET,
    "R@": WordType.F_R_PEEP,
    "RECURSE": WordType.F_RECURSE,
    "REPEAT": WordType.F_REPEAT,
    "ROT": WordType.F_ROT,
    "RSHIFT": WordType.F_RSHIFT,
    'S"': WordType.F_S_QUOTE,
    "S>D": WordType.F_S_TO_D,
    "SIGN": WordType.F_SIGN,
    "SM/REM": WordType.F_SM_SLASH_REM,
    "SOURCE": WordType.F_SOURCE,
    "SPACE": WordType.F_SPACE,
    "SPACES": WordType.F_SPACES,
    "STATE": WordType.F_STATE,
    "SWAP": WordType.F_SWAP,
    "THEN": WordType.F_THEN,
    "TYPE": WordType.F_TYPE,
    "U.": WordType.F_U_DOT,
    "U<": WordType.F_U_LESS,
    "UM*": WordType.F_UM_TIMES,
    "UM/MOD": WordType.F_UM_SLASH_MOD,
    "UNLOOP": WordType.F_UNLOOP,
    "UNTIL": WordType.F_UNTIL,
    "VARIABLE": WordType.F_VARIABLE,
    "WHILE": WordType.F_WHILE,
    "WORD": WordType.F_WORD,
    "XOR": WordType.F_XOR,
    "[": WordType.F_LEFT_BRACKET,
    "[']": WordType.F_BRACKET_TICK,
    "[CHAR]": WordType.F_BRACKET_CHAR,
    "]": WordType.F_RIGHT_BRACKET,
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

    src_words = src.upper().strip().split()
    word_rec = []
    for word in src_words:
        if word.isnumeric():
            word_rec.append(Word(WordType.NUMBER, word))
        elif word in WORD_TAG_DICT:
            word_rec.append(Word(WORD_TAG_DICT[word], word))
        else:
            word_rec.append(Word(WordType.IDENT, word))
    word_rec.append(Word(WordType.EOF, "<EOF>"))
    return word_rec
