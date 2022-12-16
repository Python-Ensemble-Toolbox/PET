import unittest

from input_output.read_config import read_clean_file, remove_empty_lines, parse_keywords


class TestPiptInit(unittest.TestCase):
    """
    Test core methods in read_txt() which parser .pipt files.
    """
    def setUp(self):
        # Read "parser_input.pipt" and parse with core methods in read_txt
        lines = read_clean_file('tests/parser_input.pipt')
        clean_lines = remove_empty_lines(lines)
        self.keys = parse_keywords(clean_lines)

    def test_single_input(self):
        # String
        self.assertIsInstance(self.keys['keyword1'], str)
        self.assertEqual(self.keys['keyword1'], 'string1')

        # Float
        self.assertIsInstance(self.keys['keyword2'], float)
        self.assertEqual(self.keys['keyword2'], 1.0)

    def test_multiple_input_single_row(self):
        # String
        self.assertIsInstance(self.keys['keyword3'], list)
        self.assertListEqual(self.keys['keyword3'], ['string1', 'string2', 'string3'])

        # Float
        self.assertIsInstance(self.keys['keyword4'], list)
        self.assertListEqual(self.keys['keyword4'], [1.0, 2.0, 3.0, 4.0])

    def test_multiple_input_multiple_rows(self):
        # String
        self.assertIsInstance(self.keys['keyword5'], list)
        self.assertIsInstance(self.keys['keyword5'][0], list)
        self.assertIsInstance(self.keys['keyword5'][1], list)
        self.assertListEqual(self.keys['keyword5'], [['string1', 'string2'], ['string3']])

        # Float
        self.assertIsInstance(self.keys['keyword6'], list)
        self.assertIsInstance(self.keys['keyword6'][0], list)
        self.assertIsInstance(self.keys['keyword6'][1], list)
        self.assertListEqual(self.keys['keyword6'], [[1.0, 2.0, 3.0], [4.0, 5.0]])

    def test_combinations_single_row(self):
        # Combination of strings and floats
        self.assertIsInstance(self.keys['keyword7'], list)
        self.assertIsInstance(self.keys['keyword7'][0], str)
        self.assertIsInstance(self.keys['keyword7'][1], list)
        self.assertIsInstance(self.keys['keyword7'][2], str)

        self.assertListEqual(self.keys['keyword7'], ['string1', [1.0, 2.0], 'string3'])

    def test_combinations_multiple_rows(self):
        # Combination of strings and floats
        self.assertIsInstance(self.keys['keyword8'], list)
        self.assertIsInstance(self.keys['keyword8'][0], list)
        self.assertIsInstance(self.keys['keyword8'][0][0], str)
        self.assertIsInstance(self.keys['keyword8'][0][1], list)
        self.assertIsInstance(self.keys['keyword8'][0][2], list)
        self.assertIsInstance(self.keys['keyword8'][0][3], str)
        self.assertIsInstance(self.keys['keyword8'][0][4], float)
        self.assertIsInstance(self.keys['keyword8'][1], list)
        self.assertIsInstance(self.keys['keyword8'][1][0], str)
        self.assertIsInstance(self.keys['keyword8'][1][1], list)

        self.assertListEqual(self.keys['keyword8'], [['string1', [1.0, 2.0], [3.0, 4.0], 'string2', 5.0], 
            ['string3', [6.0, 7.0, 8.0]]])

    def test_string_without_tab(self):
        # String with whitespaces instead of \t are parsed as single string
        self.assertIsInstance(self.keys['keyword9'], str)
        self.assertEqual(self.keys['keyword9'], 'string1 string2')
    