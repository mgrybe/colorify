import torch
from torch import nn
import torch.optim as optim
from loss import GANLoss


class MainModel(nn.Module):
    def __init__(self, net_G=None, net_D=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)

        self.gan_criterion = GANLoss().to(self.device)
        self.criterion = nn.L1Loss().to(self.device)

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data[0].to(self.device)
        self.ab = data[1].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.gan_criterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.gan_criterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, no_gan=False):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_L1 = self.criterion(fake_image, torch.cat([self.L, self.ab], dim=1)) * self.lambda_L1
        self.loss_G = self.loss_G_L1

        if not no_gan:
            self.loss_G_GAN = self.gan_criterion(fake_preds, True)
            self.loss_G += self.loss_G_GAN

        self.loss_G.backward()

    def optimize(self, train_D=True, train_G=True, no_gan=False):
        if train_D:
            self.forward()
            self.net_D.train()
            self.set_requires_grad(self.net_D, True)
            self.opt_D.zero_grad()
            self.backward_D()
            self.opt_D.step()
        if train_G:
            self.net_G.train()
            self.set_requires_grad(self.net_D, False)
            self.opt_G.zero_grad()
            self.backward_G(no_gan)
            self.opt_G.step()

if __name__ == '__main__':
    from generator import Generator
    from discriminator import PatchDiscriminator
    model = MainModel(Generator(), PatchDiscriminator(3))
    model.setup_input((torch.rand(16, 1, 256, 256), torch.rand(16, 2, 256, 256)))
    model()