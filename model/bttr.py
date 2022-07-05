from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from bttr.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


class BTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,  # 编码器维度
        growth_rate: int,  # 编码器densenet卷积层通道数
        num_layers: int,  # 一个block内卷积层个数
        nhead: int,  # 多头注意力和自注意力
        num_decoder_layers: int,  # 解码器层数
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )  # encoder的输出为img:（batch_size, h*w, d_model）, img_mask:(batch_size, h*w)
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)
        # dubug这个tgt，观察pad、sos、eos，之前没找到其实现
        out = self.decoder(feature, mask, tgt)

        return out

    def beam_search(
        self, img: FloatTensor, img_mask: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [1, 1, h', w']
        img_mask: LongTensor
            [1, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]  因为在val和test期间，batch_size为1
        return self.decoder.beam_search(feature, mask, beam_size, max_len)
