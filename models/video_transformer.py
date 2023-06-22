import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import argparse

from models.transformer import (
    TransformerDecoderLayer,
    TransformerDecoder,
    PositionalEncoderLike,
)
from config import CONFIG


class VideoTransformerBase(nn.Module):
    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VideoTransformerQueryDecoder(VideoTransformerBase):
    """
    TQN-style video decoder.
    Injests learnable queries self.query_embed (that correspond to positions of steps)
    and attends to the video to produce a sequence of steps.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        num_queries,
        hidden_dim=1024,
        d_output=None,
        hidden_dropout=0.1,
        output_dropout=0.3,
        use_feature_pos_enc=True,
    ):
        super().__init__()

        # Decoder params
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.d_output = d_output
        self.use_feature_pos_enc = use_feature_pos_enc

        # Decoder Layers
        decoder_layer = TransformerDecoderLayer(
            self.d_model, self.num_heads, hidden_dim, hidden_dropout, "relu", normalize_before=True
        )
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm, return_intermediate=False)

        # Learnable Queries
        self.query_embed = nn.Embedding(self.num_queries, self.d_model)

        # Decoder output
        if d_output is not None:
            self.output_dropout = nn.Dropout(output_dropout)
            self.output_projection = nn.Linear(d_model, d_output)

        # Positional encoding for features
        if use_feature_pos_enc:
            self.pos_enc_like = PositionalEncoderLike(d_model, max_seq_len=CONFIG.DATASET.MAX_VIDEO_LEN)

        # inititialize weights
        self.apply(self._init_weights)

    def forward(self, features, features_mask=None):
        """Query Decoder"""
        B, T, D = features.shape
        features = features.transpose(0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        tgt = torch.zeros_like(query_embed)
        pos = self.pos_enc_like(features) if hasattr(self, "pos_enc_like") else None
        out = self.decoder(tgt, features, memory_key_padding_mask=features_mask, pos=pos, query_pos=query_embed)
        out = out.squeeze(0).transpose(0, 1)
        if self.d_output is not None:
            out = self.output_dropout(out)
            out = self.output_projection(out)
        return out

    def inference(self, features, features_mask=None):
        # makikng predictions
        activations = self.forward(features, features_mask)
        predictions = activations.argmax(-1)

        # formating the predicitons nicely
        padded_predictions = torch.zeros_like(predictions).fill_(-2)
        padded_pred_mask = torch.zeros_like(predictions).to(bool).fill_(True)
        for b in range(predictions.size(0)):
            eos_idxs = (predictions[b] == CONFIG.DATASET.NUM_CLASSES).nonzero(as_tuple=False)
            eos_idx = CONFIG.DATASET.MAX_STEPS + 1 if eos_idxs.numel() == 0 else eos_idxs[0] + 1
            padded_predictions[b, :eos_idx] = predictions[b, :eos_idx]
            padded_pred_mask[b, :eos_idx] = False
        return padded_predictions, padded_pred_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--featD", type=int, default=768, help="feature dimension")
    parser.add_argument("--numDecHeads", type=int, default=8, help="num Decoder Heads")
    parser.add_argument("--numDecLayers", type=int, default=2, help="num Decoder layers")
    parser.add_argument("--numEncHeads", type=int, default=4, help="num Encoder Heads")
    parser.add_argument("--numEncLayers", type=int, default=2, help="num Encoder layers")
    parser.add_argument("--numProts", type=int, default=16, help="number of prototypes/queries")

    args = parser.parse_args()
    enc = VideoTransformerEncoder(d_model=args.featD, num_heads=args.numEncHeads, num_layers=args.numEncLayers)
    dec = VideoTransformerDecoder(
        d_model=args.featD,
        num_heads=args.numDecHeads,
        num_layers=args.numDecLayers,
        num_queries=args.numProts,
        d_output=105,
    )
    features = torch.rand((8, 300, 768))  # [batch,seq_len,featD]
    enc_out = enc(features)
    print(features.shape)
    print(enc_out.shape)
    dec_out = dec(features)
    print(dec_out.shape)
