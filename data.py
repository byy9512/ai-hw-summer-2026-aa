from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_dataloaders(data_dir: str = "./data", batch_size: int = 128, num_workers: int = 2, augment: bool = False):
    """Returns (train_loader, test_loader) for MNIST, downloading to data_dir if needed.

    When augment=True, the train split gets random rotation, translation/scale, and
    random erasing. The test split is always left unaugmented (only normalized), so
    evaluation stays a fair, fixed comparison across runs.
    """
    normalize = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        train_transform = test_transform

    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
