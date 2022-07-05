import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from bttr.datamodule.vocab import CROHMEVocab

vocab = CROHMEVocab()

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory

# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])  # 对所有的样本按图片大小升序排列
    # 一个batch可以包含更多的小图片，包含较少的大图片，值得学习

    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = transforms.ToTensor()(fea)
        if size > biggest_image_size:
            biggest_image_size = size  # 维护最大的图片尺寸
        batch_image_size = biggest_image_size * (i + 1)  # 每个batch使用统一的大小，保证不超过最大的图片
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")  # 预设最大的标签长度为200，超出的忽略
        elif size > maxImagesize:  # 一个超参，根据GPU的能力来设置，超出的样本忽略
            print(
                f"image: {fname} size: {fea.shape[1]} x {fea.shape[2]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data(archive: ZipFile, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"{dir_name}/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data  # 返回的是一个列表，每个元素是一个元组（文件名，图片，公式）


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1  # 意为一个batch的数据，包含batch_size条样本
    batch = batch[0]  # 取到一个batch中的元组，再下一级便是文件名、图片、标签组成的列表
    fnames = batch[0]  # 每个
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]  # 把标签全部转化成索引

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)  # 保存下样本数
    max_height_x = max(heights_x)  # 一个batch内的最大高度
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)  # 1是为了和原图片的三通道一致
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)  # 二通道
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x  # x张量存下一个batch内的所有图片
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0  # x_mask保存一个图片的实际有效位位置

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y)  # 使用自己创建的数据格式


def build_dataset(archive, folder: str, batch_size: int):
    data = extract_data(archive, folder)  # 文件名、图片、公式
    return data_iterator(data, batch_size)  # 返回按batch_size指定的batch数据，与data三个元素相同


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        test_year: str = "2014",
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)  # 判断类型是否一致，相较于type函数考虑了类之间的继承
        # assert表达式可以让程序在崩溃之前就抛出异常，打断程序执行
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = build_dataset(archive, "train", self.batch_size)
                # batch_size的大小实际上是变化的，首先是8，之后依次降序
                self.val_dataset = build_dataset(archive, self.test_year, 1)
            if stage == "test" or stage is None:
                self.test_dataset = build_dataset(archive, self.test_year, 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        # 经过dataloader后，batchsize仍然是一个可以变化的，最大值为指定的batchsize大小

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    batch_size = 8

    parser = ArgumentParser()
    parser = CROHMEDatamodule.add_argparse_args(parser)

    args = parser.parse_args(["--batch_size", f"{batch_size}"])

    dm = CROHMEDatamodule(**vars(args))
    dm.setup()
    # print(batch_size)
    train_loader = dm.train_dataloader()
    print(type(train_loader))
    for i in train_loader:
        print(i)
