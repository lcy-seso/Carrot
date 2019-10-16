from __future__ import print_function

import os
import shutil
import unittest

from context import dataset
import dataset.ptb


class TestPtb(unittest.TestCase):
    def setUp(self):
        self.dict = dataset.ptb.get_vocab(min_word_freq=3)

    def test_ptb_vocab(self):
        dict_len_cut3 = 9932

        self.assertEqual(len(self.dict), dict_len_cut3)
        self.assertEqual(self.dict['<unk>'], len(self.dict) - 1)

    def test_train(self):
        N = 0  # the first line.
        line = (
            u"aer banknote berlitz calloway centrust cluett fromstein gitano "
            u"guterman hydro-quebec ipo kia memotec mlx nahb punts "
            u"rake regatta rubens sim snack-food ssangyong swapo wachter <e>")
        line_ids = [
            self.dict.get(w, self.dict['<unk>']) for w in line.strip().split()
        ]

        x, _ = dataset.ptb.train(max_length=len(line_ids), min_word_freq=3)
        self.assertEqual(line_ids, x[N].tolist())

    def test_test(self):
        N = 0  # the first line.
        line = u" no it was n't black monday <e>"
        line_ids = [
            self.dict.get(w, self.dict['<unk>']) for w in line.strip().split()
        ]

        x, _ = dataset.ptb.test(max_length=len(line_ids), min_word_freq=3)
        self.assertEqual(line_ids, x[N].tolist())

    def test_valid(self):
        N = 0  # the first line.
        line = (u" consumers may want to move their telephones "
                u"a little closer to the tv set <e>")
        line_ids = [
            self.dict.get(w, self.dict['<unk>']) for w in line.strip().split()
        ]

        x, _ = dataset.ptb.valid(max_length=len(line_ids), min_word_freq=3)
        self.assertEqual(line_ids, x[N].tolist())


if __name__ == '__main__':
    unittest.main()
