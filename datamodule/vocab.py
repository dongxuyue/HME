import os
from functools import lru_cache
from typing import Dict, List


@lru_cache()
def default_dict():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")


class CROHMEVocab:

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, dict_path: str = default_dict()) -> None:
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX

        with open(dict_path, "r") as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)  # 将vocab生成字典，给出索引

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}  # 通过数字索引可以找到字符

        # print(f"Init vocab with size: {len(self.word2idx)}")
    # 以下定义了三个方法，分别是词列表转索引列表，索引列表转词列表，以及索引列表直接转标签字符串
    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)
