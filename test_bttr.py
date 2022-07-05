# 作者：买旭旭不用券
from zipfile import ZipFile

from PIL import Image

from bttr.datamodule.datamodule import Data
from bttr.datamodule.datamodule import *

batch_size = 8

with ZipFile('./data.zip') as archive:

    train_dataset = build_dataset(archive, "train", batch_size)

for item in train_dataset:
    names, pics, labels = item
    print(len(labels))