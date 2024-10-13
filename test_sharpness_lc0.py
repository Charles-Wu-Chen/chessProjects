import unittest
import numpy as np
from functions import sharpnessLC0

class TestSharpnessLC0(unittest.TestCase):
    def test_normal_cases(self):
        self.assertAlmostEqual(sharpnessLC0([300, 400, 300]), 1.3929, places=4)
        self.assertEqual(sharpnessLC0([100, 800, 100]), 0.2071)

    def test_edge_cases(self):
        self.assertAlmostEqual(sharpnessLC0([1, 998, 1]), 0.0210, places=4)
        self.assertEqual(sharpnessLC0([0, 1000, 0]), 0.0118)

    def test_asymmetric_cases(self):
        self.assertAlmostEqual(sharpnessLC0([600, 100, 300]), 20.4901, places=4)
        self.assertAlmostEqual(sharpnessLC0([200, 300, 500]), 2.0814, places=4)
        self.assertAlmostEqual(sharpnessLC0([850, 100, 50]), 2.7328, places=4)
        self.assertAlmostEqual(sharpnessLC0([527, 372, 101]), 2.7328, places=4)


    def test_extreme_values(self):
        self.assertEqual(sharpnessLC0([1000, 0, 0]), 4000000.0000)
        self.assertEqual(sharpnessLC0([0, 0, 1000]), 4000000.0000)

    def test_invalid_input(self):
        with self.assertRaises(IndexError):
            sharpnessLC0([500, 500])
        with self.assertRaises(TypeError):
            sharpnessLC0("not a list")

    def test_specific_chess_positions(self):
        # 1.d4 move WDL is [463, 142, 395]
        self.assertEqual(sharpnessLC0([463, 142, 395]), 12.1146)

        # 2.e4 move WDL is [476, 134, 389]
        self.assertAlmostEqual(sharpnessLC0([476, 134, 389]), 13.3397, places=4)

        # 3r2k1/4pp1p/2R3p1/b3B3/6P1/4P2P/5K2/8 w - - 0 1 WDL [218, 382, 400]
        self.assertEqual(sharpnessLC0([218, 382, 400]), 1.4125)

        # rnb1kb1r/1p1n1pp1/p3p2p/4q3/3NN2B/4Q3/2P3PP/3RKB1R w Kkq - 0 16 WDL [430, 96, 474]
        self.assertEqual(sharpnessLC0([430, 96, 474]), 26.854)

if __name__ == '__main__':
    unittest.main()