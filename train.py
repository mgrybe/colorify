import torch

from generator import build_generator, init_model
from discriminator import PatchDiscriminator
from gan import MainModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = build_generator()
disc = init_model(PatchDiscriminator(3), device)
model_gan = MainModel(net_G=gen, net_D=disc, lambda_L1=100, lr_G=0.0002, lr_D=0.0002, beta1=0.5, beta2=0.999).to(device)
