import torch

from discriminator import PatchDiscriminator
from utils import ReceptiveFieldCalculator

discriminator = PatchDiscriminator(3)

if __name__ == '__main__':
    calculator = ReceptiveFieldCalculator()
    calculator.calculate({
        'l1': [4, 2, 1],  # kernel, stride, padding
        'l2': [4, 2, 1],
        'l3': [4, 2, 1],
        'l4': [4, 1, 1],
        'l5': [4, 1, 1]
    }, 256)

if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256)
    output = discriminator(input)
    print(f'output.shape={output.shape}')
    print(discriminator)
