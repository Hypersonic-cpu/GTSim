"""Verify compiler prefetch pages against independently enumerated offsets."""

import unittest

from parser import ast as A


class TileLayoutTests(unittest.TestCase):
    def test_physical_prefetch_coverage(self):
        layout = A.TileMajorLayout(16, 128, 32)
        layout.validate((64, 256))
        for row, col, height, width in (
            (0, 0, 16, 64),
            (16, 64, 16, 64),
            (48, 0, 16, 256),
            (31, 127, 2, 2),
            (0, 0, 64, 256),
        ):
            with self.subTest(row=row, col=col, height=height, width=width):
                expected = set()
                for k in range(row, row + height):
                    for n in range(col, col + width):
                        offset = (
                            (k // 32) * 32 * 256 * 2
                            + (n // 128) * 8192
                            + ((k % 32) // 16) * 4096
                            + 2 * ((n % 128) * 16 + k % 16)
                        )
                        expected.add(offset // 4096)
                self.assertEqual(
                    set(layout.pages(256, row, col, height, width)), expected
                )

    def test_no_partial_expert_layout(self):
        with self.assertRaises(ValueError):
            A.TileMajorLayout(16, 128, 32).validate((48, 256))


if __name__ == "__main__":
    unittest.main()
