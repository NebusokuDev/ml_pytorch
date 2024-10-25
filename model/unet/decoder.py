from torch.nn import Module


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, skip_input, prev_decoder_input):
        pass
