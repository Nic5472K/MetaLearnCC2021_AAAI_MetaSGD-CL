###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Supporting script, tests base learner on all tasks
#       see Az02 for base learner model
#       see Az01 for test loader
#       see A001 for task-wise permutation

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
def My_testAz02(model, test_loader, All_Perm):

    My_TN = len(All_Perm)
    #---
    model.eval()
    ResultsDict = {}

    print('')
    print('+++'*5)

    for TaskID in range(My_TN):

        #---
        My_Perm = All_Perm[TaskID]

        #---
        correct_tot = 0
        tot_dat     = 0
        
        for data, target in test_loader:

            #---
            data    = data.view(data.shape[0], -1).cuda()
            data    = data[:, My_Perm]
            logits  = model(data)

            #---
            _, y_hat    = torch.max(
                            logits.cpu(),
                            1, keepdim=False
                            )
            
            correct_tot += (target == y_hat).float().sum()
            tot_dat     += data.shape[0]

        print('Tested task {}/{}'.format(TaskID + 1, My_TN))

        acc = correct_tot / tot_dat
        ResultsDict[TaskID] = acc.numpy()

    print("")

    return ResultsDict
