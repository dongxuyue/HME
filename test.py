from pytorch_lightning import Trainer
from typing import List, Optional
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from bttr.datamodule.datamodule import data_iterator, collate_fn
from bttr.lit_bttr import LitBTTR

test_year = "2014"
ckp_path = r"D:\A文档\A学习用资料\research\毕设\实验结果\res50-adadelta\epoch=93-step=141093-val_ExpRate=0.4447.ckpt"





class test_module(pl.LightningDataModule):
    def __init__(
        self,
        img_path: str ,
    ) -> None:
        super().__init__()
        # assert表达式可以让程序在崩溃之前就抛出异常，打断程序执行
        self.imgfile_path = img_path
        print(f"Load data from: {self.imgfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        data = []

        img = Image.open(self.imgfile_path).copy()
        img_name = self.imgfile_path.split('/')[-1]
        formula = []
        data.append((img_name, img, formula))
        self.test_dataset = data_iterator(data, batch_size=1)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers= 0,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    trainer = Trainer(logger=False, gpus=1)

    # dm = CROHMEDatamodule(test_year=test_year)
    dm = test_module(r'D:\A文档\A学习用资料\research\毕设\代码\ResBttr\BTTR-main\learn-pyqt\learn-designer\test.bmp')

    model = LitBTTR.load_from_checkpoint(ckp_path)

    a = trainer.test(model, datamodule=dm)



# if __name__ == '__main__':
#     test = test_module('D:/A文档/A学习用资料/research/毕设/代码/ResBttr/BTTR-main/data/2019/ISICal19_1201_em_752.bmp')
#     test.setup()
#     test_loader = test.test_dataloader()
#     print(type(test_loader))
#     for i in test_loader:
#         print(i)
