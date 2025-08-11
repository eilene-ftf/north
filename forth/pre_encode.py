"""Pre-encode the syntax of programs."""

from lex import lex

if __name__ == "__main__":
    with open("./examples/sample.forth", "r") as f:
        buff = f.read()
        print(buff)

    words = lex(buff)
    print(words)
