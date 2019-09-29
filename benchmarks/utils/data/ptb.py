import io

from torchtext import data

file = {
    "train": "ptb.train.txt",
    "valid": "ptb.valid.txt",
    "test": "ptb.test.txt",
    "vocab": "ptb.vocab.txt",
    "root": "data"
}

UNK = "<unk>"
PAD = "</p>"
BOS = "<s>"
EOS = "<e>"

TEXT = data.Field(lower=True, pad_token=PAD, unk_token=UNK)


class LanguageModelingDataset(data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, text_field, encoding='utf-8', **kwargs):
        """Create a LanguageModelingDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('target', text_field)]
        examples = []

        with io.open(path, encoding=encoding) as f:
            for line in f:
                preprocessed_text = [BOS] + text_field.preprocess(line) + [EOS]
                text = preprocessed_text[:-1]
                target = preprocessed_text[1:]
                examples.append(data.Example.fromlist([text, target], fields))
        super(LanguageModelingDataset, self).__init__(examples, fields,
                                                      **kwargs)


class PennTreebank(LanguageModelingDataset):
    """The Penn Treebank dataset.
    A relatively small dataset originally created for POS tagging.
    References
    ----------
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank
    """

    urls = [
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
    ]
    name = 'penn-treebank'
    dirname = ''

    @classmethod
    def splits(cls,
               text_field,
               root='.data',
               train=file['train'],
               validation=file['valid'],
               test=file['test'],
               **kwargs):
        """Create dataset objects for splits of the Penn Treebank dataset.
        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory where the data files will be stored.
            train: The filename of the train data. Default: 'ptb.train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'ptb.valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'ptb.test.txt'.
        """
        return super(PennTreebank, cls).splits(
            root=root,
            train=train,
            validation=validation,
            test=test,
            text_field=text_field,
            **kwargs)


train_dataset, valid_dataset, test_dataset = PennTreebank.splits(TEXT)
TEXT.build_vocab(train_dataset)

vocab = TEXT.vocab
