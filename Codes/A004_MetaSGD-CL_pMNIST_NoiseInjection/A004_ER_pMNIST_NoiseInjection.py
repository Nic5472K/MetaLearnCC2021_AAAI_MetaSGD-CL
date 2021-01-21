###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Training permuted MNIST with ER with
#   ring buffer size 1000
#   but with noise injection
#   in the form of randomly shuffling some pixels of the images
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
    EM_c    = int(1000 / My_TN)
    coTrain = 10

    NI = 0.5

    train_loader, test_loader = LoaderAz01(My_BS)

    MyModel     = MLP()
    MyModel     = MyModel.cuda()

    criterion   = nn.CrossEntropyLoss()
    optimiser   = optim.SGD(MyModel.parameters(), lr = 0.01)

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
    print('Mode: Single')
    print('')

    print("+++"*5)
    print('Testing the randomly initialised base learner')
    PreTrained_Results = My_testAz02(MyModel, test_loader, All_Perm)

    print("=-="*5)
    print("Pre-training results")
    print("---"*5)
    print("|| Target \t || Accuracy")
    for TaskID in range(0, My_TN):
        CurrentID   = list(PreTrained_Results.keys())[TaskID]
        CurrentACC  = PreTrained_Results[CurrentID]
        print("|| {} \t || {}%".\
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

    for TaskID in range(My_TN):

        temp_x = []
        temp_y = []
        
        print('---' * 5)
        print('A new task has started')
        print('')

        for batch_idx, (data, target) in enumerate(train_loader):   
                
            data    = data.view(My_BS, -1).cuda()
            target  = target.cuda()

            if (batch_idx + 1) <= int(EM_c / My_BS):
                temp_x.append(data)
                temp_y.append(target)

            My_Perm     = All_Perm[TaskID]

            #<><><>#
            # noise injection
            NoiseLoc = np.random.choice(784, int(784*NI), replace = False)

            My_Perm_candidate = torch.tensor(My_Perm)[NoiseLoc]
            random.shuffle(My_Perm_candidate.numpy())

            My_PermxXx = torch.tensor(My_Perm)
            My_PermxXx[NoiseLoc] = My_Perm_candidate

            data        = data[:, My_PermxXx]

            if TaskID > 0:
                new_data   = []
                new_target = []

                ClassCoTrain = \
                    np.ceil(
                        (F.softmax(
                            torch.rand(
                                TaskID),
                            dim = 0) * coTrain).numpy())

                for PastID in range(TaskID):
                    EM_x = ER_EM_x[PastID]
                    EM_y = ER_EM_y[PastID]

                    if TaskID == 1:
                        CurClassCoTrain = coTrain
                    else:
                        CurClassCoTrain = int(ClassCoTrain[PastID])

                    coTrainLoc = np.random.choice(EM_c, CurClassCoTrain,
                                                  replace = False)

                    Cur_x = EM_x[coTrainLoc]
                    Cur_y = EM_y[coTrainLoc]

                    Cur_Perm    = All_Perm[PastID]

                    #<><><>#
                    # noise injection
                    NoiseLoc = np.random.choice(784, int(784*NI), replace = False)

                    Cur_Perm_candidate = torch.tensor(Cur_Perm)[NoiseLoc]
                    random.shuffle(Cur_Perm_candidate.numpy())

                    Cur_PermxXx = torch.tensor(Cur_Perm)
                    Cur_PermxXx[NoiseLoc] = Cur_Perm_candidate

                    Cur_x       = Cur_x[:, Cur_PermxXx]

                    new_data.append(  Cur_x)
                    new_target.append(Cur_y)

                new_data   = torch.cat(new_data,   dim = 0)
                new_target = torch.cat(new_target, dim = 0).long()

                data   = torch.cat((data,   new_data),   dim = 0)
                target = torch.cat((target, new_target), dim = 0)
                    
            MyModel.train()
            MyModel.zero_grad()
            y_hat = MyModel(data)

            loss = criterion(y_hat, target)
            loss.backward()
            optimiser.step()

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
        EoTE_Accuracy = My_testAz02(MyModel, test_loader, All_Perm)

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
        print('ER ringbuffer TotMem1000 NI 50%')
        print('Seed: {}'.format(seed))
        print('--'*20)
        for itr in range(My_TN + 1):
            cur = [round(i, 2) for i in ARiT[itr,:].numpy()]

            print_cur = ''
            for itr2 in cur:
                print_cur += str(itr2) + ' '

            print(print_cur)



                



            


