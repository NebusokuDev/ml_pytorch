from trainer.unet_trainer import UNetTrainer
from model.unet.unet import UNet

unet = UNet()
trainer = UNetTrainer(unet)

trainer.build_model("unet", epoch=60)