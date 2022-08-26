from torch.nn.utils.rnn import pad_sequence


class FastSpeech(nn.Module):
    def __init__(self, config: FastSpeechConfig, device: torch.device):
        super().__init__()
        self.embedding_size = config.embedding_size
        self.device = device
        self.encoder = FastSpeechEncoder(config.vocab_size, config.embedding_size,
                                         config.n_heads, config.dropout,
                                         config.hidden_channel, config.n_blocks)

        self.length_regulator = LengthRegulator(config.embedding_size,
                                                config.dropout,
                                                config.sr,
                                                config.hop_size,
                                                config.alpha,
                                                device)

        self.decoder = FastSpeechDecoder(config.embedding_size, config.n_heads,
                                         config.dropout, config.hidden_channel,
                                         config.mel_size, config.n_blocks)

    def forward(self, x: torch.Tensor, durations=None):
        """
        Args:
            x: Tensor, shape [batch_size, MaxInLen (in this batch)]
        """

        out = self.encoder(x)

        pred_durations = self.length_regulator(out)

        if not self.training:
            durations = pred_durations

        durations = durations.clone().detach().type(torch.LongTensor).to(self.device)

        res = []
        for i in range(out.shape[0]):
            curr_elem = torch.repeat_interleave(out[i], durations[i], dim=0)
            res.append(curr_elem)

        out = pad_sequence(res, batch_first=True)
        out = self.decoder(out)
        out = torch.transpose(out, 1, 2)
        return out, pred_durations
