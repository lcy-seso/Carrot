from torch.nn import Embedding, Linear, Module
from torch.nn.init import uniform_, zeros_


class LanguageModel(Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, lstm, vocab_size: int, input_size: int,
                 hidden_size: int, bidirectional: bool):
        super(LanguageModel, self).__init__()
        self.encoder = Embedding(vocab_size, input_size)
        self.lstm = lstm(input_size, hidden_size, bidirectional=bidirectional)
        if bidirectional:
            self.decoder = Linear(2 * hidden_size, vocab_size)
        else:
            self.decoder = Linear(hidden_size, vocab_size)

        weight_range = 0.1
        uniform_(self.encoder.weight, -weight_range, weight_range)
        uniform_(self.decoder.weight, -weight_range, weight_range)
        zeros_(self.decoder.bias)

    def forward(self, input):
        # shape of `input` is (seq_len, batch_size)
        embedding = self.encoder(input)
        # shape of `embedding` is (seq_len, batch_size, embedding_size)
        output, hidden = self.lstm(embedding)
        # shape of `output` is (seq_len, batch_size, hidden_size)
        decoded = self.decoder(output)
        return decoded, hidden
