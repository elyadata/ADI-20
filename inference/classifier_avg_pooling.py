import torch

from speechbrain.inference.interfaces import Pretrained


class WhisperDialectClassifierWithAvgPooling(Pretrained):
    MODULES_NEEDED = [
        "whisper",
        "output_mlp"
    ]

    def encode_batch(self, wavs, wav_lens=None, return_full_embeddings: bool = False):
        """Encodes the input audio into a single vector embedding.
        The waveforms should already be in the model's desired format.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        embeddings = self.mods.whisper(wavs, wav_lens)
        if return_full_embeddings:
            return embeddings
        else:
            return self.hparams.avg_pool(embeddings)

    def classify_batch(self, wavs, wav_lens=None):
        """Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        embeddings = self.encode_batch(wavs, wav_lens, return_full_embeddings=False)
        embeddings = self.mods.output_mlp(embeddings)
        out_prob = self.hparams.log_softmax(embeddings)
        score, index = torch.max(out_prob, dim=-1)
        self.hparams.label_encoder.expect_len(self.hparams.n_languages)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def classify_file(self, path, **kwargs):
        """Classifies the given audiofile into the given set of labels.

        Arguments
        ---------
        path : str
            Path to audio file to classify.
        **kwargs : dict
            Arguments forwarded to ``load_audio``.

        Returns
        -------
        out_prob : torch.Tensor
            The log posterior probabilities of each class ([batch, N_class])
        score : torch.Tensor
            It is the value of the log-posterior for the best class ([batch,])
        index : torch.Tensor
            The indexes of the best class ([batch,])
        text_lab : list of str
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        waveform = self.load_audio(path, **kwargs)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        return self.classify_batch(wavs=batch, wav_lens=rel_length)

    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.classify_batch(wavs, wav_lens)
