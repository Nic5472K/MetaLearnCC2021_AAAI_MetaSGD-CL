###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Training permuted MNIST with MetaSGD-CL but with
#   ring buffer and only 100 memory
#   See a version of the commented codes in A001
#   Find scripts Az01 - Az05 in File A001

###===###
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
from    Az01_Loader         import *
from    Az02_BaseLearner    import *
from    Az03_Eval           import *
from    Az04_UpdateProgress import *
from    Az05_ERMetaSGD      import RefMode, ER_MSGD

###===###
for SITR in range(0, 20):
    seed = SITR
    random.seed(                seed)
    np.random.seed(             seed)
    torch.manual_seed(          seed)
    torch.cuda.manual_seed(     seed)
        
    My_BS   = 10
    MyObs   = 1000
    MyOpT   = int(MyObs / My_BS)
    My_TN   = 10
    # the difference is here, changed from 250 for each task
    # to 100, and this is for all tasks
    EM_c    = int(100 / My_TN)
    coTrain = 10

    train_loader, test_loader = LoaderAz01(My_BS)

    RefModel    = MLP()
    RefModel    = RefModel.cuda()
    Optimiser   = ER_MSGD(
                    RefMode(RefModel)
                    )
    Optimiser   = Optimiser.cuda()

    All_Perm    = []
    for itr in range(My_TN):
        base_perm = [jtr for jtr in range(784)]
        if itr == 0:
            All_Perm.append(base_perm)
        else:
            random.shuffle(base_perm)
            All_Perm.append(base_perm)

    print('=='* 20)
    print('Permuted MNIST with 2-L MLP')
    print('Mode: MetaSGD-CL')

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

    Old_Results     = PreTrained_Results
    Best_Results    = PreTrained_Results
    Straight_After  = []
    Straight_Before = []

    ARiT = torch.zeros(My_TN + 1, My_TN)
    Initial_Results = PreTrained_Results.copy()
    Ragain = np.array(list(Initial_Results.values()))
    Ragain = [round(i * 100, 2) for i in Ragain]
    ARiT[0, :] = torch.tensor(Ragain)

    ER_EM_x = torch.zeros(My_TN, EM_c, 784).cuda()
    ER_EM_y = torch.zeros(My_TN, EM_c).cuda()

    MyModel = MLP()
    MyModel = MyModel.cuda()

    criterion   = nn.CrossEntropyLoss()

    for TaskID in range(My_TN):

        Optimiser.UpdateMetaLR(TaskID)
        OOO = optim.Adam(Optimiser.parameters(), lr=1e-3)

        temp_x = []
        temp_y = []
        
        print('---' * 5)
        print('A new task has started')
        print('')

        for batch_idx, (data, target) in enumerate(train_loader):

            Optimiser.Unroll(model=MyModel)

            data    = data.view(My_BS, -1).cuda()
            target  = target.cuda()  

            if (batch_idx + 1) <= int(EM_c / My_BS):
                temp_x.append(data)
                temp_y.append(target)

            My_Perm     = All_Perm[TaskID]
            data        = data[:, My_Perm]

            new_grad = {}
            if TaskID > 0:
                new_data    = []
                new_target  = []

                for PastID in range(TaskID):

                    EM_x = ER_EM_x[PastID]
                    EM_y = ER_EM_y[PastID]
                    coTrainLoc = np.random.choice(EM_c, coTrain,
                                                  replace = False)

                    Cur_x = EM_x[coTrainLoc]
                    Cur_y = EM_y[coTrainLoc].long()
                    Cur_Perm    = All_Perm[PastID]
                    Cur_x       = Cur_x[:, Cur_Perm]

                    new_data.append(  Cur_x)
                    new_target.append(Cur_y)

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

                new_data   = torch.cat(new_data,   dim = 0)
                new_target = torch.cat(new_target, dim = 0).long()

                Rdata   = torch.cat((data,   new_data),   dim = 0)
                Rtarget = torch.cat((target, new_target), dim = 0)                    

            MyModel.train()
            MyModel.zero_grad()
            y_hat = MyModel(data)

            loss = criterion(y_hat, target)
            loss.backward()

            Optimiser.train()
            Optimiser.zero_grad()
            RefModel = Optimiser.UpdateTransfer(MyModel,
                                                TaskID,
                                                new_grad,
                                                My_BS,
                                                coTrain)

            if TaskID == 0:
                RefModel.train()
                RefModel.zero_grad()
                f_x      = RefModel(data)
                loss = criterion(f_x, target)

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

                loss.backward()
                for param in Optimiser.parameters():
                    try:
                        param.grad.data.clamp_(-1, 1)
                    except:
                        continue
                OOO.step()

            if np.mod( (batch_idx + 1), int( MyOpT / 10)) == 0:
                print('Progress: {}%'.\
                      format(
                          round( (batch_idx + 1) / MyOpT * 100, 2 )
                          )
                      )

            if batch_idx + 1 == MyOpT:

                temp_x = torch.cat(temp_x, 0)
                temp_y = torch.cat(temp_y, 0)

                ER_EM_x[TaskID] = temp_x
                ER_EM_y[TaskID] = temp_y

                break

        print("")
        print("+++"*5)
        print('EoTE')
        EoTE_Accuracy = My_testAz02(RefModel, test_loader, All_Perm)

        Ragain = np.array(list(EoTE_Accuracy.values()))
        Ragain = [round(i * 100, 2) for i in Ragain]
        ARiT[TaskID + 1, :] = torch.tensor(Ragain)

        Straight_After, Straight_Before, Old_Results, Best_Results = \
                        UPAz04(
                            TaskID, My_TN,
                            EoTE_Accuracy,
                            Straight_After, Straight_Before,
                            Old_Results, Best_Results
                            )

    if 1:
        print('--'*20)
        for itr in range(My_TN + 1):
            cur = [round(i, 2) for i in ARiT[itr,:].numpy()]

            print_cur = ''
            for itr2 in cur:
                print_cur += str(itr2) + ' '

            print(print_cur)




                


