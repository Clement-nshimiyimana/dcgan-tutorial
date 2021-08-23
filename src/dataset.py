
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
try:
    from .config import args
except ModuleNotFoundError:
    from config import args

# We can use an image folder dataset the way we have it setup.
# Create the dataset

dataset = dset.ImageFolder(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)