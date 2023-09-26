import torch
from torch import nn
from torch.nn import functional as F


class Mask(nn.Module):
    def __init__(self, n_branches, module):
        super(Mask, self).__init__()
        self.n_branches = n_branches
        self.module = module

    def forward(self, mask_matrices): #weight, feature
        """ mask_matrices = []
        for i in range(self.n_branches):
            #print(weight[i].shape, feature[i].shape) # torch.Size([4, 512]) torch.Size([4, 512, 243, 17, 2])
            temp = self.CAM(weight[i], feature[i])

            #print(temp.shape) # torch.Size([243, 17, 2])
            mask_matrices.append(temp) """
        
        # mask_matrices = list of three tensors of shape N, T, J, M

        for branch_attn in mask_matrices:
            N, M, T, J = branch_attn.shape
            print("mask in shape: " , branch_attn.shape)
            branch_attn = branch_attn.view(-1)
            branch_attn = F.softmax(branch_attn, dim=0)
            branch_attn[branch_attn > 0.3] = 1
            branch_attn[branch_attn <= 0.3] = 0
            branch_attn = 1 - branch_attn
            branch_attn = branch_attn.view(N, M, T, J)

        for i in range(1, self.n_branches):
            for j in range(i):
                if j == 0:
                    mask = mask_matrices[j]
                else:
                    mask *= mask_matrices[j]
            #mask = torch.cat([mask.unsqueeze(1)] * 4, dim=1)
            print("mask: ", mask.shape) # torch.Size([243, 4, 17, 2])
            #mask = mask.permute(1,0,2,3)
            self.module.masks[i].data = mask.detach()#
            #print(mask.view(-1).shape) # torch.Size([33048])
    
    def CAM(self, weight, feature):
        N, C = weight.shape
        weight = weight.view(N, C, 1, 1, 1).expand_as(feature)
        result = (weight * feature).sum(dim=1)
        result = result.mean(dim=0) # averaging activations across samples of the batch to get one mask per pathway

        T, V, M = result.shape
        result = result.view(-1)
        result = 1 - F.softmax(result, dim=0)
        result[result > 0.3] = 1
        result[result <= 0.3] = 0
        result = result.view(T, V, M)
        return result

