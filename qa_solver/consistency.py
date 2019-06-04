# %%
import random
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm


# %%
class PredictRelations(Dataset):
    def __init__(self, path='../../fast_table.pth.tar'):
        super().__init__()

        data = torch.load(path)

        self.tuples = data['rows']
        self.table = dict(zip(data['elements'], data['vecs']))

    def convert(self, word):
        return torch.from_numpy(self.table[word])

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        sub, rel, obj = self.tuples[idx]

        x = torch.cat([self.convert(sub), self.convert(obj)], -1)
        y = self.convert(rel)

        return x, y


# %%
def iterate(mode, model, data_loader, device, criterion, optim, loss_type):
    if mode == 'train':
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loader = tqdm(enumerate(data_loader), total=len(data_loader))

    dists = []

    for i, (x, y) in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        if loss_type == 'cos':
            loss = criterion(pred, y, torch.tensor(1.).to(device))
        else:
            loss = criterion(pred, y).mean(-1)

        dists.append(loss.detach())
        loss = loss.mean()

        if mode == 'train':
            optim.zero_grad()
            loss.backward()
            optim.step()

        loader.set_description("loss: {:.4f}".format(loss.item()))

    del loader

    torch.set_grad_enabled(True)

    return torch.cat(dists).cpu()


# %%
###############################
# Train Model
###############################
class SubsetDataset(Dataset):

    def __init__(self, data_source, indices):
        self.data_source = data_source
        self.indices = indices

    def __getitem__(self, idx):
        return self.data_source[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def simple_split_dataset(dataset, split, shuffle=True):
    """

    :param dataset:
    :param split: split percent as ratio [0,1]
    :param shuffle:
    :return:
    """

    assert 0 < split < 1

    ncut = int(len(dataset) * split)

    part1 = SubsetDataset(dataset, np.arange(0, ncut))
    part2 = SubsetDataset(dataset, np.arange(ncut, len(dataset)))

    return part1, part2


# %%
def train_helper(relations_path, dim, device, val_per, num_workers, batch_size, lr, weight_decay, loss_type,
                 num_epochs):
    dataset = PredictRelations(relations_path)

    nonlin = nn.PReLU
    model = nn.Sequential(
        nn.Linear(dim * 2, 512),
        nonlin(),
        nn.Linear(512, dim),
    )

    model.to(device)
    print(model)

    train_data, val_data = simple_split_dataset(dataset, shuffle=False, split=1 - val_per)
    train_loader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_type == 'cos':
        criterion = nn.CosineEmbeddingLoss(reduction='none')
    else:
        criterion = nn.MSELoss(reduction='none')

    for epoch in range(num_epochs):
        train_stats = iterate('train', model, train_loader, device, criterion, optim, loss_type)
        val_stats = iterate('val', model, val_loader, device, criterion, optim, loss_type)
        print(f'Epoch {epoch + 1}, Train: {train_stats.mean().item():.4f}, Val: {val_stats.mean().item():.4f}')


# %%
def random_search():
    static_params = {
        "relations_path": "../data-nb/fast_table.pth.tar",
        "device": "cuda:0",
        "num_epochs": 10,
        "loss_type": "mse",
        "num_workers": 8,
        "val_per": 0.1,
        "dim": 100,
        "batch_size": 128
    }

    variable_params = {
        # "dim": [100, 200, 300],
        # "batch_size": [64, 128],
        "lr": [1e-1, 1e-2, 1e-3, 1e-4],
        "weight_decay": [1e-1, 1e-2, 1e-3, 1e-4]
    }

    # Run random search
    used_variable_params = set()
    param_space_size = np.prod([len(p) for p in variable_params])
    while len(used_variable_params) < param_space_size:
        curr_params = {k: random.choice(v) for k, v in variable_params.items()}
        curr_variable_params = tuple(chain.from_iterable(curr_params.items()))

        if curr_variable_params not in used_variable_params:
            print("Current variable parameters:", curr_variable_params)
            used_variable_params.add(curr_variable_params)
            curr_params.update(static_params)
            train_helper(**curr_params)


random_search()
