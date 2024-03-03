import torch
import torch.nn.functional as F

def sim(z1, z2):
    # 归一化 dim=32*64
    z1 = F.normalize(z1)
    # dim=32*64
    z2 = F.normalize(z2)
    return (z1 * z2).sum(dim=1)


def INFONCELoss(z1, z2):
    f = lambda x: torch.exp(x / 0.7)
    between_sim = f(sim(z1, z2))
    rand_item = torch.randperm(z1.shape[0])
    neg_sim = f(sim(z1, z2[rand_item])) + f(sim(z2, z1[rand_item]))
    return -torch.log(between_sim / (between_sim + between_sim + neg_sim)).mean()