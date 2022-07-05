from typing import List, Tuple

import editdistance
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric

from bttr.datamodule import vocab


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('error1', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('error2', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[int], indices: List[int]):
        dist = editdistance.eval(indices_hat, indices)  # 使用编辑距离计算字符识别率

        if dist == 0:  # 如果完全匹配，准确率加1
            self.rec += 1
        elif dist == 1:
            self.error1 += 1
        elif dist == 2:
            self.error2 += 1

        self.total_line += 1  # 总数加一

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line  # 计算返回exprate
        one_error = self.error1/self.total_line
        return exp_rate


class One_errorRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('error1', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[int], indices: List[int]):
        dist = editdistance.eval(indices_hat, indices)  # 使用编辑距离计算字符识别率

        if dist <= 1:
            self.error1 += 1

        self.total_line += 1  # 总数加一

    def compute(self) -> float:
        one_error = self.error1 / self.total_line
        return one_error


class Two_errorRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('error2', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[int], indices: List[int]):
        dist = editdistance.eval(indices_hat, indices)  # 使用编辑距离计算字符识别率

        if dist <= 2:
            self.error2 += 1

        self.total_line += 1  # 总数加一

    def compute(self) -> float:
        two_error = self.error2 / self.total_line
        return two_error


class SymRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sym", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[int], indices: List[int]):
        dist = editdistance.eval(indices_hat, indices)  # 使用编辑距离计算字符识别率
        self.sym += (len(indices_hat)-dist)
        self.total_line += len(indices_hat)  # 总数加一

    def compute(self) -> float:
        sym_rate = self.sym / self.total_line  # 计算返回exprate
        return sym_rate


def ce_loss(
    output_hat: torch.Tensor, output: torch.Tensor, ignore_idx: int = vocab.PAD_IDX
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx)
    return loss


def to_tgt_output(
    tokens: List[List[int]], direction: str, device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    direction : str
        one of "l2r" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]  # 实现一个句子中各个token的反转
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]  # 保存一个batch内所有句子的真实长度
    tgt = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)  # tgt是只包含开始标志，out只包含结束标志
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    # (SOS, 1, 2, 3, 4)
    # (EOS, 4, 3, 2, 1)
    out = torch.cat((l2r_out, r2l_out), dim=0)
    # (1, 2, 3, 4, EOS)
    # (4, 3, 2, 1, SOS)
    return tgt, out
