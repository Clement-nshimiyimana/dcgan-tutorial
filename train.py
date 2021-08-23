"""Main entrypoint to train dcgan"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import args
from src.models import Generator, Discriminator
from src.utils import weights_init
from src.utils import train_dcgan
from src.dataset import dataloader


def main(args):
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")


    # Create the generator
    netG = Generator(args.ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)


    # Create the Discriminator
    netD = Discriminator(args.ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    train_dcgan(netD, netG, criterion, optimizerD, optimizerG, dataloader, 
          args.num_epochs, device, real_label, fake_label, fixed_noise, args.nz)


if __name__ == "__main__":
    main(args)