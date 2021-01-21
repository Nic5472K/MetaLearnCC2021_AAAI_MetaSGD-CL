###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Supporting script, prepares a MLP base learner
#       layers              = 2;
#       input dimension     = 784 (28 x 28);
#       hidden dimension    = 100; &
#       output dimension    = 10

import  random
import  math
import  numpy                   as      np

import  torch
import  torch.nn                as      nn
import  torch.optim             as      optim
import  torch.nn.functional     as      F

import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms

###===###
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        HD = 100

        self.L1 = nn.Linear(784, HD)
        self.L2 = nn.Linear(HD, HD)
        self.L3 = nn.Linear(HD, 10)

        self.init_weight()

        self.count_param()

    def init_weight(self):

        for param in self.parameters():
            if len(param.shape) == 1:
                param.data.copy_(
                    torch.zeros_like(param))
            else:
                std = 1.0 * math.sqrt(
                                2.0 / sum( list(param.shape) )
                                )
                a   = math.sqrt(3.0) * std
                param.data.copy_(
                    (torch.rand_like(param) * 2 - 1) * a
                    )

    def count_param(self):

        tot_P = 0

        for param in self.parameters():
            tot_P += param.shape.numel()

        print('Total param: {}'.format(tot_P))
        
        
    def forward(self, x):
        
        x0 = x.view(-1, 784)
        x1 = torch.relu(self.L1(x0))
        x2 = torch.relu(self.L2(x1))
        x3 = self.L3(x2)

        y_hat = F.log_softmax(x3, dim = 1)

        return y_hat
