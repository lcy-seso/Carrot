from __future__ import print_function

import os
import shutil
import unittest

from context import dataset
import dataset.ptb


class TestPtb(unittest.TestCase):
    def setUp(self):
        self.dict = dataset.ptb.get_vocab(min_word_freq=None)

    def test_ptb_vocab(self):

        self.assertEqual(len(self.dict), 10001)
        self.assertEqual(self.dict['<unk>'], len(self.dict))

    def test_train(self):
        N = 0
        line = (
            "aer banknote berlitz calloway centrust cluett "
            "fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb "
            "punts rake regatta rubens sim snack-food ssangyong swapo wachter")
        line_ids = [
            self.dict.get(ch, self.dict['<unk>'])
            for ch in line.strip().split()
        ]

        read_line = dataset.ptb.train()[N]
        self.assertEqual(line_ids, read_line)

    def test_test(self):
        N = 10
        UNK = self.dict['<unk>']

        line = ("<unk> james <unk> chairman of specialists henderson brothers "
                "inc. it is easy to say the specialist is n't doing his job")
        line_ids = [self.dict.get(ch, UNK) for ch in line.strip().split()]

        read_line = dataset.ptb.test()[N]
        self.assertEqual(line_ids, read_line)

    def test_valid(self):
        N = 6
        UNK = self.dict['<unk>']

        line = (" but right now programmers are figuring that viewers who are "
                "busy dialing up a range of services may put down their <unk> "
                "control <unk> and stay <unk>")
        line_ids = [self.dict.get(ch, UNK) for ch in line.strip().split()]

        read_line = dataset.ptb.valid()[N]
        self.assertEqual(line_ids, read_line)


if __name__ == '__main__':
    unittest.main()
