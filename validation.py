from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

ckp_path = r"D:\A文档\A学习用资料\research\毕设\实验结果\res50-adadelta\epoch=93-step=141093-val_ExpRate=0.4447.ckpt"
test_year = "2014"

# model

dm = CROHMEDatamodule(test_year=test_year)
dm.setup(stage='fit')

# train
trainer = Trainer(gpus=1)
model = LitBTTR.load_from_checkpoint(ckp_path)

# test using the best model!
trainer.validate(model=model, datamodule=dm)