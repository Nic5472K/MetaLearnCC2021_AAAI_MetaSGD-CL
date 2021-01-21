###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Training permuted MNIST with MetaSGD-CL

###===###
# standard
import  random
import  math
import  numpy                   as      np
#---
# torch related
import  torch
import  torch.nn                as      nn
import  torch.optim             as      optim
import  torch.nn.functional     as      F
import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms
#---
# personal codes
from    Az01_Loader         import *
from    Az02_BaseLearner    import *
from    Az03_Eval           import *
from    Az04_UpdateProgress import *
#---
from    Az05_ERMetaSGD      import RefMode, ER_MSGD

###===###
# SITR: Seed ITRation
for SITR in range(0, 20):
    ###===###
    # set seeds
    #---
    seed = SITR
    random.seed(                seed)
    np.random.seed(             seed)
    torch.manual_seed(          seed)
    torch.cuda.manual_seed(     seed)
        
    ###===###
    # set hyperparameters
    #---
    # batch size
    My_BS   = 10
    # the amount of training data per task
    MyObs   = 1000
    # the amount of batches per task
    MyOpT   = int(MyObs / My_BS)
    # the amount of tasks in total
    My_TN   = 10
    # the amount of memory storage (hard storage)
    EM_c    = 250
    # the amount of co-training samples from each past storage
    coTrain = 10

    ###===###
    # setup the loader
    train_loader, test_loader = LoaderAz01(My_BS)

    ###===###
    # setup MetaSGD-CL before the CL meta-training phase
    # create a reference model to be saved in
    RefModel    = MLP()
    RefModel    = RefModel.cuda()
    # our meta-learner neural optimiser
    Optimiser   = ER_MSGD(
                    RefMode(RefModel)
                    )
    Optimiser   = Optimiser.cuda()

    ###===###
    # create the permutation masks for each pMNIST task
    All_Perm    = []
    for itr in range(My_TN):
        base_perm = [jtr for jtr in range(784)]
        if itr == 0:
            All_Perm.append(base_perm)
        else:
            random.shuffle(base_perm)
            All_Perm.append(base_perm)

    ###===###
    # just some introduction text on your screen
    print('=='* 20)
    print('Permuted MNIST with 2-L MLP')
    print('Mode: MetaSGD-CL')

    ####===###
    # here we test the performance of the randomly initialised reference model
    print("+++"*5)
    print('Testing the randomly initialised base learner')
    PreTrained_Results = My_testAz02(RefModel, test_loader, All_Perm)
    print("=-="*5)
    print("Pre-training results")
    print("---"*5)
    print("|| Target \t || Accuracy")
    for TaskID in range(0, My_TN):
        CurrentID   = list(PreTrained_Results.keys())[TaskID]
        CurrentACC  = PreTrained_Results[CurrentID]
        print("|| {} \t\t || {}%".\
              format( CurrentID + 1,
                      round(CurrentACC * 100, 2)
                      )
              )
    print("")

    ###===###
    # create storage for UPAz04 in Az04_UpdateProgress
    Old_Results     = PreTrained_Results
    Best_Results    = PreTrained_Results
    Straight_After  = []
    Straight_Before = []
    # this is matrix R in the paper
    # useful for computing all metric values
    ARiT = torch.zeros(My_TN + 1, My_TN)
    Initial_Results = PreTrained_Results.copy()
    Ragain = np.array(list(Initial_Results.values()))
    Ragain = [round(i * 100, 2) for i in Ragain]
    ARiT[0, :] = torch.tensor(Ragain)

    ###===###
    # here we create the memory for MetaSGD-CL
    ER_EM_x = torch.zeros(My_TN, EM_c, 784).cuda()
    ER_EM_y = torch.zeros(My_TN, EM_c).cuda()

    ###===###
    # define our base learner
    MyModel = MLP()
    MyModel = MyModel.cuda()
    # and criterion
    criterion   = nn.CrossEntropyLoss()

    ###===###
    # let us begin the CL phase
    # for every task
    for TaskID in range(My_TN):
        ###===###
        # When starting a new task,
        # create new learning rates \beta
        Optimiser.UpdateMetaLR(TaskID)
        # define Optimiser of Optimiser (OOO)
        # for the newly introduced \beta
        OOO = optim.Adam(Optimiser.parameters(), lr=1e-3)

        ###===###
        # create temporal storage for updating the
        # episodic memory
        temp_x = []
        temp_y = []
        
        ###===###
        print('---' * 5)
        print('A new task has started')
        print('')
        # for each mini-batch
        for batch_idx, (data, target) in enumerate(train_loader):
            ###===###
            # unroll the MetaSGD-CL like in L2LbGDbGD
            # to remove the reference model from the computation graph
            Optimiser.Unroll(model=MyModel)

            ###===###
            data    = data.view(My_BS, -1).cuda()
            target  = target.cuda()  

            ###===###
            # fill the storage up
            if (batch_idx + 1) <= int(EM_c / My_BS):
                temp_x.append(data)
                temp_y.append(target)

            ###===###
            # apply permutation to make MNIST into pMNIST
            My_Perm     = All_Perm[TaskID]
            data        = data[:, My_Perm]

            ###===###
            # now we need to compute the new gradient
            # of the old tasks
            new_grad = {}
            if TaskID > 0:
                new_data    = []
                new_target  = []
                # for every past task
                for PastID in range(TaskID):
                    # find their corresponding data and label
                    EM_x = ER_EM_x[PastID]
                    EM_y = ER_EM_y[PastID]
                    # randomly find what data to use
                    coTrainLoc = np.random.choice(EM_c, coTrain,
                                                  replace = False)

                    Cur_x = EM_x[coTrainLoc]
                    Cur_y = EM_y[coTrainLoc].long()
                    # and remember to apply the correct permutations
                    Cur_Perm    = All_Perm[PastID]
                    Cur_x       = Cur_x[:, Cur_Perm]

                    new_data.append(  Cur_x)
                    new_target.append(Cur_y)

                    ###===###
                    # now feed the data in and get the corresponding gradient
                    MyModel.train()
                    MyModel.zero_grad()
                    y_hat = MyModel(Cur_x)

                    loss = criterion(y_hat, Cur_y)
                    loss.backward()

                    Local_Grads = []
                    
                    for module in MyModel.children():
                        Local_Grads.append(
                            module._parameters['weight'].grad.data.view(-1))
                        Local_Grads.append(
                            module._parameters['bias'].grad.data.view(-1))

                    Local_Grads = torch.cat(Local_Grads, dim = 0)

                    new_grad[PastID + 1] = Local_Grads

                ###===###
                # aggregate all data,
                # later used for updating the meta-learner
                new_data   = torch.cat(new_data,   dim = 0)
                new_target = torch.cat(new_target, dim = 0).long()

                Rdata   = torch.cat((data,   new_data),   dim = 0)
                Rtarget = torch.cat((target, new_target), dim = 0)                    
                    
            ###===###
            # now we return to the new task
            MyModel.train()
            MyModel.zero_grad()
            y_hat = MyModel(data)

            loss = criterion(y_hat, target)
            loss.backward()

            ###===###
            # update the base learner
            # and store the updated weights into the reference model
            Optimiser.train()
            Optimiser.zero_grad()
            RefModel = Optimiser.UpdateTransfer(MyModel,
                                                TaskID,
                                                new_grad,
                                                My_BS,
                                                coTrain)

            ###==##
            # now update the meta-learner neural optimiser
            # the following if-else is only differs on
            # the usage of (data, target) vs (Rdata, Rtarget)
            #---
            # MetaSGD-CL is updated by
            # "how much more can the updated reference model be further updated"
            # see Equation (12) in our paper
            if TaskID == 0:
                RefModel.train()
                RefModel.zero_grad()
                f_x      = RefModel(data)
                loss = criterion(f_x, target)

                ###===###
                loss.backward()
                for param in Optimiser.parameters():
                    try:
                        param.grad.data.clamp_(-1, 1)
                    except:
                        continue
                OOO.step()                
            else:
                RefModel.train()
                RefModel.zero_grad()
                f_x      = RefModel(Rdata)
                loss = criterion(f_x, Rtarget)

                ###===###
                loss.backward()
                for param in Optimiser.parameters():
                    try:
                        param.grad.data.clamp_(-1, 1)
                    except:
                        continue
                OOO.step()
            
            ###===###
            # the progress of each meta-training cycle
            if np.mod( (batch_idx + 1), int( MyOpT / 10)) == 0:
                print('Progress: {}%'.\
                      format(
                          round( (batch_idx + 1) / MyOpT * 100, 2 )
                          )
                      )
            ###===###
            # don't exceed the agreed amount of viewed observations
            if batch_idx + 1 == MyOpT:
                ###===###
                # and update the memory
                temp_x = torch.cat(temp_x, 0)
                temp_y = torch.cat(temp_y, 0)

                ER_EM_x[TaskID] = temp_x
                ER_EM_y[TaskID] = temp_y

                ###===###
                break

        #---
        # End of task evaluation (EoTE)
        print("")
        print("+++"*5)
        print('EoTE')
        EoTE_Accuracy = My_testAz02(RefModel, test_loader, All_Perm)

        #---
        Ragain = np.array(list(EoTE_Accuracy.values()))
        Ragain = [round(i * 100, 2) for i in Ragain]
        ARiT[TaskID + 1, :] = torch.tensor(Ragain)

        #---
        Straight_After, Straight_Before, Old_Results, Best_Results = \
                        UPAz04(
                            TaskID, My_TN,
                            EoTE_Accuracy,
                            Straight_After, Straight_Before,
                            Old_Results, Best_Results
                            )

    ###===###
    if 1:
        print('--'*20)
        for itr in range(My_TN + 1):
            cur = [round(i, 2) for i in ARiT[itr,:].numpy()]

            print_cur = ''
            for itr2 in cur:
                print_cur += str(itr2) + ' '

            print(print_cur)




                


