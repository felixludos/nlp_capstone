# %%
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# %%
args = Namespace()

args.kb_embedding_path = 'data-nb/fast_table.pth.tar'
args.generated_embedding_path = 'data-nb/oie_table.pth.tar'

args.dim = 300
args.device = 'cuda:0'

args.batch_size = 1024
args.num_workers = 4

args.loss_type = 'mse'


# %%
class Embedding(Dataset):
    def __init__(self, path):
        super().__init__()
        data = torch.load(path)
        self.tuples = data['rows']
        self.table = dict(zip(data['elements'], data['vecs']))

    def convert(self, word):
        return torch.from_numpy(self.table[word])

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return torch.cat([self.convert(w) for w in self.tuples[idx]], -1)


# %%
kb_dataset = Embedding(args.kb_embedding_path)
kb_data_loader = DataLoader(kb_dataset,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False)
len(kb_data_loader)
# %%
generated_dataset = Embedding(args.generated_embedding_path)
generated_data_loader = DataLoader(generated_dataset,
                                   num_workers=0,
                                   batch_size=256,
                                   shuffle=False)
len(generated_data_loader)


# %%
def cosd(x, y):
    if x.ndimension() == 1:
        x = x.unsqueeze(0)
    if y.ndimension() == 1:
        y = y.unsqueeze(0)
    x = F.normalize(x, 2, -1)
    y = F.normalize(y, 2, -1)
    return -x @ y.transpose(-1, -2) / 2 + .5


def l2(x, y):
    if x.ndimension() == 1:
        x = x.unsqueeze(0)
    if y.ndimension() == 1:
        y = y.unsqueeze(0)
    x = x.unsqueeze(-2)
    y = y.unsqueeze(-3)
    return (x - y).pow(2).mean(-1)


# %%
class Aggregator(nn.Module):
    def __init__(self, criterion='mean', k=10, largest=True):
        super().__init__()

        self.crit = criterion

        self.k = k
        self.largest = largest

        self.reset()

    def reset(self):
        self.best = None
        self.N = 0

    def forward(self, cands):
        N = cands.size(-1)

        if self.crit in {'top', 'extrema'} and self.best is not None:
            cands = torch.cat([self.best, cands], -1)

        if self.crit == 'top':
            self.best = cands.topk(self.k, dim=-1, largest=self.largest)[0]
        elif self.crit == 'mean':
            cands = cands.sum(-1, keepdim=True)

            if self.best is not None:
                self.best += cands
            else:
                self.best = cands

            self.N += N

    def get_result(self):
        assert self.best is not None

        best = self.best

        if self.crit == 'mean':
            best = self.best / self.N

        return best


# %%
agg = Aggregator('top', k=10, largest=False)
# agg = Aggregator('mean')

dist_fn = cosd

outs = []
for i, generated_embedding in enumerate(tqdm(generated_data_loader)):
    agg.reset()

    for kb_embedding in kb_data_loader:
        dist = dist_fn(generated_embedding, kb_embedding)
        agg(dist)

    outs.append(agg.get_result())
    # print('{}/{} : {} {}'.format(i+1,len(xloader), len(outs), outs[-1].shape))

outs = torch.cat(outs, 0)
outs.shape
# %%
losses = outs.mean(-1)
# %%
# torch.save(losses, 'oie_overlap.pth.tar')
# %%
plt.hist(losses.numpy(), bins=120)
# %%
# Print statistics
idx = losses.argsort().numpy()
options = np.array(generated_dataset.tuples)

k = 100

print('top {}'.format(k))
for o in options[idx[:k]]:
    print('{:<15} {:<15} {:<15}'.format(*o))

print('bottom {}'.format(k))
for o in options[idx[-k:]]:
    print('{:<15} {:<15} {:<15}'.format(*o))
