# Image Colorization with Pretrained U-Net (ResNet18) and GAN

This project implements an image colorization model using a U-Net architecture with a ResNet18 backbone, combined with GAN-based techniques. The solution incorporates insights from the following repositories:

- [Image Colorization with U-Net and GAN Tutorial](https://github.com/mberkay0/image-colorization)
- [DeOldify](https://github.com/jantic/DeOldify)

## Generator

The generator is a U-Net model with a pretrained ResNet18 backbone, chosen to provide robust feature extraction capabilities, which aids in producing more accurate and detailed colorization.

## Discriminator

The discriminator employs a "Patch" discriminator structure, similar to the model used in the [Image Colorization with U-Net and GAN Tutorial](https://github.com/mberkay0/image-colorization). This implementation enhances the standard design by including spectral normalization and a Self-Attention block, both of which contribute to improved model stability and finer detail preservation in the colorized images.

## Training

This project follows a NoGAN training approach, as introduced by [DeOldify](https://github.com/jantic/DeOldify). In NoGAN training, the generator and discriminator are trained separately, allowing stable generator training before engaging adversarial learning. After this phase, a brief period of GAN training is applied to fine-tune the results, improving realism in the colorized outputs.

## Dataset

This project uses a subset of 10,000 images from the [Flickr dataset](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL), which were resized to 256x256 pixels for training and evaluation. This dataset provides a diverse range of subjects, helping the model generalize effectively across different types of images.

## Results

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LbNR_IwiWm5XTPpLSo754aS9W-Lp2sA-?usp=sharing)

- **Left**: Images trained only using MAE (L1) loss.
- **Right**: Images trained with 1 epoch of actual GAN training.
- Each triplet shows: grayscale input, predicted colorization, and original image for comparison.

<table>
    <tr>
    <td>
        <img src="/results/l1/image_6.png">
    </td>
    <td>
        <img src="/results/gan/image_0e_6.png">
    </td>
    </tr>
    <tr>
    <td>
        <img src="/results/l1/image_14.png">
    </td>
    <td>
         <img src="/results/gan/image_0e_14.png">
    </td>
    </tr>
    <tr>
    <td>
        <img src="/results/l1/image_16.png">
    </td>
    <td>
         <img src="/results/gan/image_0e_16.png">
    </td>
    </tr>
    <tr>
    <td>
        <img src="/results/l1/image_23.png">
    </td>
    <td>
         <img src="/results/gan/image_0e_23.png">
    </td>
    </tr>
    <tr>
    <td>
        <img src="/results/l1/image_34.png">
    </td>
    <td>
        <img src="/results/gan/image_0e_34.png">
    </td>
    </tr>
</table>
