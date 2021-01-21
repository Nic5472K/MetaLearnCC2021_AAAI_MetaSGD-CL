###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Supporting script, the MetaSGD-CL meta-learner nueral optimiser

###===###
import  numpy               as      np
import  math
from    functools           import reduce
from    operator            import mul
#---
import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      Fnc
import  torch.optim         as      optim
from    torch.autograd      import  Variable

###===###
# ER_MSGD
#   Experience Replay with Meta-SGD
class ER_MSGD(nn.Module):

    ###===###
    def __init__(self, model):
        super(ER_MSGD, self).__init__()
        
        ###===###
        # keep a reference of the base line model for clean coding
        self.RefModel = model

        ###===###
        # save a copy of the total parametric count
        Tot_P_Count = 0
        for params in self.RefModel.model.parameters():
            Tot_P_Count += params.shape.numel()

        #---
        self.Tot_P_Count = Tot_P_Count

    ###===###
    # unroll the neural optimiser
    # to prepare for updating the neural optimiser
    # unlike the LSTM-optimiser of L2L by gradient descent by gd
    # MetaSGD is unrolled once every single iteration
    # note, this unroll occurs before a new mini-batch is fed
    def Unroll(self, model=None):
        # to remove the old reference off the graph
        self.RefModel.reset()
        # and update its most recent weights
        self.RefModel.copy_params_from(model)     

    ###===###
    # creates new meta learning rates
    def UpdateMetaLR(self, TaskID):
        # whenever encountering a new task
        # create a new learning rate per parameter
        # initialised with value 0.01
        newLR = torch.ones(self.Tot_P_Count).cuda() * 0.01
        setattr(self,
                'LR_Task_{}'.format(TaskID),
                nn.Parameter(newLR)
                )
        # if the task at hand is not the first task (task 0)
        if TaskID > 0:
            # remove the old task lr using .data
            oldLR = getattr(self, 'LR_Task_{}'.format(TaskID - 1)).cpu().detach().data
            oldLR = torch.tensor(oldLR).cuda()
            
            exec('del self.LR_Task_{}'.format(TaskID - 1))
            
            setattr(self,
                    'LR_Task_{}'.format(TaskID - 1),
                    oldLR
                    )

    ###===###
    # the main code for using MetaSGD to update the base learner
    def UpdateTransfer(self,
                       CurOptimisee,
                       TaskID,
                       new_grad,
                       My_BS,
                       coTrain):
        
        ###===###
        # grab all the native gradients 
        grads = []
        for module in CurOptimisee.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))            

        ###===###
        # grab all the existing parametric values
        flat_params = self.RefModel.get_flat_params()

        ###===###
        # see Equation (9) in our paper where
        # \theta_{t+1} =
        #   \theta_t - \beta_V L_V
        #            - \sum \beta_u \L_u
        #---
        # let us first execute the
        # ( \theta_t - \beta_V L_V ) part
        # grab the current lr
        CUR_LR   = getattr(self, 'LR_Task_{}'.format(TaskID))
        # and compare it with an upper bound kappa (see Equation (10) )
        CUR_LR   = torch.min(CUR_LR, torch.ones_like(CUR_LR) * 0.02)
        # the first part of the update
        flat_params = flat_params - CUR_LR * torch.cat(grads)      
        #---
        # the remaining part of the update
        # (          - \sum \beta_u \L_u )
        # for all other observed tasks
        if TaskID > 0:
            # go into each past task
            for PastID in range(TaskID):
                # look for the past trained and frozen learning rates
                cur_LR   = getattr(self, 'LR_Task_{}'.format(PastID + 1))
                # look for their corresponding gradient
                cur_grad = new_grad[PastID + 1]
                # compare with kappa (see Equation (10) )
                # and prevent over-parametrisation (see Equation (11))
                cur_LR   = torch.min(cur_LR, torch.ones_like(cur_LR) * 0.02) / (TaskID + 1)
                # the second part of the update
                flat_params = flat_params - cur_LR * cur_grad

        ###===###
        # now copy the updated parameters back to the non-referential learner
        self.RefModel.set_flat_params(flat_params)
        self.RefModel.copy_params_to(CurOptimisee)
        
        return self.RefModel.model      

###===###
# a nice set of utility functions
class RefMode:

    def __init__(self, model):
        self.model = model
        
    def reset(self):
        
        for module in self.model.children():
            if isinstance(module, nn.Linear):
                module._parameters['weight'] = Variable(
                    module._parameters['weight'].data)
                module._parameters['bias'] = Variable(
                    module._parameters['bias'].data)
            if isinstance(module, nn.Conv2d):
                module._parameters['weight'] = Variable(
                    module._parameters['weight'].data)                

    def get_flat_params(self):
        params = []

        for module in self.model.children():
            if isinstance(module, nn.Linear):
                params.append(module._parameters['weight'].view(-1))
                params.append(module._parameters['bias'].view(-1))
            if isinstance(module, nn.Conv2d):
                params.append(module._parameters['weight'].view(-1))                

        return torch.cat(params)

    def set_flat_params(self, flat_params):

        offset = 0

        for i, module in enumerate(self.model.children()):
            if isinstance(module, nn.Linear):
                weight_shape = module._parameters['weight'].size()
                bias_shape = module._parameters['bias'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)
                bias_flat_size = reduce(mul, bias_shape, 1)

                module._parameters['weight'] = flat_params[
                    offset:offset + weight_flat_size].view(*weight_shape)
                module._parameters['bias'] = flat_params[
                    offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)

                offset += weight_flat_size + bias_flat_size
                
            if isinstance(module, nn.Conv2d):
                
                weight_shape = module._parameters['weight'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)

                module._parameters['weight'] = flat_params[
                    offset:offset + weight_flat_size].view(*weight_shape)

                offset += weight_flat_size                

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)            
