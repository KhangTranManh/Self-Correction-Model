"""Small parser checks; never execute or download dataset content."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from taco_inventory import canonical_url, io_metadata, metadata_list, normalized_text


class TacoInventoryTests(unittest.TestCase):
    def test_url_canonicalization(self):
        self.assertEqual(
            canonical_url("http://CodeForces.com/problemset/problem/1/A/?utm=x#part"),
            "https://codeforces.com/problemset/problem/1/A",
        )
        self.assertIsNone(canonical_url("file:///tmp/problem"))

    def test_io_modes(self):
        self.assertEqual(io_metadata('{"inputs":[[1]],"outputs":[2],"fn_name":"f"}'),
                         ("call_based", 1, "f"))
        self.assertEqual(io_metadata('{"inputs":["1"],"outputs":["2"]}'),
                         ("standard_input", 1, None))
        self.assertEqual(io_metadata('{"inputs":[1],"outputs":[]}')[0],
                         "unpaired_or_empty")
        self.assertEqual(io_metadata("not json")[0], "malformed")
        huge = "9" * 5000
        self.assertEqual(io_metadata('{"inputs":[' + huge + '],"outputs":[1]}')[0],
                         "standard_input")

    def test_metadata_literal_is_data_only(self):
        self.assertEqual(metadata_list("['arrays', 'sorting']"), ["arrays", "sorting"])
        self.assertIsNone(metadata_list("__import__('os').system('echo unsafe')"))
        self.assertEqual(normalized_text(" A  B\nC "), "a b c")


if __name__ == "__main__":
    unittest.main()
