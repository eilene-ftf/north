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
    F_PLUS = auto()
    F_PLUS_STORE = auto()
    F_PLUS_LOOP = auto()
    F_COMMA = auto()
    F_MINUS = auto()
    F_DOT = auto()
    F_DOT_QUOTE = auto()
    F_SLASH = auto()
    F_SLASH_MOD = auto()
    F_ZERO_LESS = auto()
    F_ZERO_EQUALS = auto()
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
    "+": WordType.F_PLUS,
    "+!": WordType.F_PLUS_STORE,
    "+LOOP": WordType.F_PLUS_LOOP,
    ",": WordType.F_COMMA,
    "-": WordType.F_MINUS,
    ".": WordType.F_DOT,
    '."': WordType.F_DOT_QUOTE,
    "/": WordType.F_SLASH,
    "/MOD": WordType.F_SLASH_MOD,
    "0<": WordType.F_ZERO_LESS,
    "0=": WordType.F_ZERO_EQUALS,
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

    src_words = src.upper().split()
    word_rec = []
    for word in src_words:
        if word.isnumeric():
            print(f"ISNUMERIC: {word}")
            word_rec.append(Word(WordType.NUMBER, word))
        if word in WORD_TAG_DICT:
            word_rec.append(Word(WORD_TAG_DICT[word], word))
        else:
            word_rec.append(Word(WordType.IDENT, word))
    word_rec.append(Word(WordType.EOF, "<EOF>"))
    return word_rec
