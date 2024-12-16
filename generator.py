import torch
from torch import nn


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super(Generator, self).__init__()

        # Encoder
        self.downscale = nn.MaxPool2d(kernel_size=2)

        self.enc1 = self.encoder_block(input_channels, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)

        # Bottleneck
        self.bottleneck = self.encoder_block(512, 1024)

        # Decoder
        self.upscale4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.decoder_block(1024, 512)
        self.upscale3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.decoder_block(512, 256)
        self.upscale2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.decoder_block(256, 128)
        self.attn1 = nn.Sequential(SelfAttention(128), nn.ReLU(inplace=True))
        self.upscale1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.decoder_block(128, 64)
        self.attn2 = nn.Sequential(SelfAttention(64), nn.ReLU(inplace=True))

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def encoder_block(self, input_channels, output_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def decoder_block(self, input_channels, output_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.downscale(enc1))
        enc3 = self.enc3(self.downscale(enc2))
        enc4 = self.enc4(self.downscale(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.downscale(enc4))

        # Decoder path
        dec4 = self.upscale4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upscale3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upscale2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.attn1(dec2)

        dec1 = self.upscale1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        dec1 = self.attn2(dec1)

        output = self.final_conv(dec1)
        return output


from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18, resnet50, resnet101
from fastai.vision.models.unet import DynamicUnet
from fastai.layers import NormType


def build_generator(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    body = create_body(resnet18(weights="DEFAULT"), pretrained=True, n_in=n_input, cut=-2)

    model = DynamicUnet(
        encoder=body,
        n_out=n_output,
        img_size=(size, size),
        norm_type=NormType.Spectral,  # Apply spectral normalization
        self_attention=True  # Add self-attention blocks
    ).to(device)

    return model


def _init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)

    return net


def init_model(model, device):
    model = model.to(device)
    model = _init_weights(model)
    return model


if __name__ == '__main__':
    #generator = Generator()
    #print(generator)
    #generator = build_generator()
    #print(generator)
    model = resnet18(weights="DEFAULT")#resnet18(weights="DEFAULT")#resnet18()
    print(model)
    #print(model.state_dict()['bn1.weight'])

    #torch.save(generator.state_dict(), f'generator-resnet18-256px-0e.pth')
