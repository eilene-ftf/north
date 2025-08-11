import unittest
import sys
import forth.lex


class TestForthLexer(unittest.TestCase):
    def test_lex(self):
        print(f"{forth.lex('> >')}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
