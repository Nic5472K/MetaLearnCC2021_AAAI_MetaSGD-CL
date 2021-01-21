###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Supporting script, keeping track of the CL progress

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
def UPAz04(
        TaskID, My_TN,
        EoTE_Accuracy,
        Straight_After, Straight_Before,
        Old_Results, Best_Results
        ):
    ###===###
    # ensures a nice layout
    #=-==-==-==-==-=
    #|| Target 	 || Current 	|| Last ....
    #---------------    

    print("=-="*5)
    p1 = "|| Target \t || Current \t"
    p2 = "|| Last \t || Best \t"
    p3 = "|| Diff2Last \t || Diff2Best"
    print(p1 + p2 + p3)
    print("---"*5)

    ###===###
    # then for every task
    for atr in range(My_TN):
        #---
        # track its ID
        CurrentID   = list(EoTE_Accuracy.keys())[atr]
        # and its current accuracy
        CurrentACC  = EoTE_Accuracy[CurrentID]

        #---
        # update the straight-after-training accuracy
        if TaskID == atr:
            Straight_After.append(CurrentACC)
            print('>>>' * 5)

        #---
        # update the previous accuracy 
        if (TaskID + 1) == atr:
            Straight_Before.append(CurrentACC)

        #---
        # find the relative accuracies
        PastAcc     = Old_Results[CurrentID]
        PrevBesAcc  = Best_Results[CurrentID]

        #---
        # updating and find the best result so far
        if CurrentACC > PrevBesAcc:
            Best_Results[CurrentID] = CurrentACC
            BestAcc = CurrentACC

        else:
            BestAcc = PrevBesAcc

        #---
        # find the relative differences between results
        CurMinusLast = CurrentACC - PastAcc
        CurMinusBest = CurrentACC - BestAcc

        #---
        # ensures a nice layout
        #|| 0 		 || 78.77% 	|| 9.18% ...
        #|| 1 		 || 10.23% 	|| 6.11% ...        
        q1 =  "|| {} \t\t || {}% \t".format(
                                        CurrentID,
                                        round(CurrentACC * 100, 2)
                                       )
        q2 = "|| {}% \t || {}% \t".format(
                                        round(PastAcc    * 100, 2),
                                        round(BestAcc    * 100, 2)
                                       )
        q3a = '{}% '.format(round(CurMinusLast  * 100, 2))
        q3b = '{}% '.format(round(CurMinusBest  * 100, 2))

        #---
        # adding a bit of status labelling between
        # the current result and the most immediate past result
        # if improved                <+>
        # if decreased (-10,    0]%  <.>
        #              (-20,  -10]%  <*>
        #              (-inf, -20]%  <**>
        if      (CurMinusLast  * 100) >  0:
                    q3a = q3a  + '<+>'
        elif -1*(CurMinusLast  * 100) > 20:
                    q3a = q3a  + '<**>'
        elif -1*(CurMinusLast  * 100) > 10:
                    q3a = q3a  + '<*>'
        elif -1*(CurMinusLast  * 100) >  0:
                    q3a = q3a  + '<.>'

        #---
        # now do the same labelling but respective to the best result
        if      (CurMinusBest  * 100) >= 0:
            #---
            # specify if it is zero-shot learning
            if atr > TaskID:
                q3b = q3b  + '<ZSL>'
            # or if it is positive baward's transfer
            else:
                q3b = q3b  + '<+>'
                
        elif -1*(CurMinusBest  * 100) > 20:
                    q3b = q3b  + '<**>'
        elif -1*(CurMinusBest  * 100) > 10:
                    q3b = q3b  + '<*>'
        elif -1*(CurMinusBest  * 100) >  0:
                    q3b = q3b  + '<.>'        

        #---
        q3 = "|| {} \t || {}".format(q3a, q3b)

        #---
        print(q1 + q2 + q3)

    #---
    # updating the old results
    Old_Results = EoTE_Accuracy
        
    print("")

    return Straight_After, Straight_Before,\
           Old_Results, Best_Results
           

           









