import os
from collections import defaultdict

import itertools

import numpy as np

from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


from lingpy import *
import xlrd

import pickle



###----------------evaluation------------------------------------------------------------



## Preparation for evaluation. First, use the imported cognate_list data. 
## output the cognate_not_dictionary. 



with open ('sca_distance.pkl', 'rb') as f:
    ipa_distance_tuple_list=pickle.load(f)


with open ('cognate_list.pickle','rb') as f:
    cognate_list=pickle.load(f)['cognate_list']



### a all_ipa_tur_ira_cross list is needed to marked the 0 and 1. 
### The format is (concept, ipa, ipa).
all_ipa_tur_ira_cross=[]

for tup in ipa_distance_tuple_list:       
    all_ipa_tur_ira_cross.append((tup[0],tup[2],tup[3]))
                  




##extract gold standard------------------------###
##cognate list contains the cognates with letters.  
cognate_not_dictionary=[]
for k,v in cognate_list.items():
    for loans in v:
        cognate_not_dictionary.append((k,loans[0][0],loans[1][0]))
    



##make note. assgin '1' if a pair is loan, assign '0' if a pair is not loan. 
def create_gold_standard_classify(tup_concept_tur_ira_in_ipa):    
    gold_standard_classify=[]
    gold_standard_cognate=[]
    for aaa in tup_concept_tur_ira_in_ipa:
        if aaa in cognate_not_dictionary:
            gold_standard_classify.append(1)
            gold_standard_cognate.append(aaa)
        else:
            gold_standard_classify.append(0)
    return gold_standard_classify,gold_standard_cognate




###########_--For evaluation--------------------------------------

##---The input of the function is a list of thresholds. 
##---For one threshold, there is one accuracy, precision, recall and f1 value. 
#-this function output the accruracy, precision, recall, and f1 list.
def eva(thres,tuple_list_concept_ipa_dis,gold):
    
    ##determing loanwords by defining a threshold. 
    predict_loanwords=[]
    predict_non_loanwords=[]
    for tt in tuple_list_concept_ipa_dis:
        if float(tt[1])<thres:
            predict_loanwords.append((tt[0],tt[2],tt[3]))
        else:
            predict_non_loanwords.append((tt[0],tt[2],tt[3]))
            
    
    ##create predict classify. aka assign 1 or 0 to the predicted loanwords. 
    predict_classifying_word_ipa=[]
    for aaa in all_ipa_tur_ira_cross:
        if aaa in predict_loanwords:
            predict_classifying_word_ipa.append(1)
        elif aaa in predict_non_loanwords:
            predict_classifying_word_ipa.append(0)
        else:
            pass
            
    #acc=accuracy_score(gold_standard_classify,predict_classifying_word_ipa)
    pre=precision_score(gold,predict_classifying_word_ipa)
    rec=recall_score(gold,predict_classifying_word_ipa)
    f1=f1_score(gold,predict_classifying_word_ipa)
    
#    if return_predict_word==True:
#        return predict_loanwords
#    elif return_only_f1==True:
#        return f1
#    else:
#        return pre,rec,f1

    return pre,rec,f1,predict_loanwords



def tuple_with_dis_to_tuple_no_dis(tuple_with_dis):
    tuple_no_dis=[]
    for t in tuple_with_dis:
        tuple_no_dis.append((t[0],t[2],t[3]))
        
    return tuple_no_dis



def average(a_list):
    return sum(a_list)/len(a_list)
    

##return the best threshold according to the training data. 
##the training data is the inputed tuple. 
##the train data includs ipa and distance. 

def best_threshold(tup, plot=False, cv=False):
    ##--make a list of distance.----------------------------####
    dis_values_list=[]
    for t in tup:
        dis_values_list.append(float(t[1]))
               
    tuple_list_concept_ipa=tuple_with_dis_to_tuple_no_dis(tup)  ## This is the tuple list used for creating gold starndard.   

 
    
    ave_distance=average(dis_values_list)    
    ## Looking for the mimimum value other than 0. 
    min_distance=min(x for x in dis_values_list if x>0) 
    
    ## There are 10 potential thresholds in case of cross evaluation.
    ## But to check the whole the dataset for once, it should be much bigger, like 200. 
    if cv==True:
        threshold=np.linspace(min_distance,ave_distance,10)
    else:
        threshold=np.linspace(min_distance,ave_distance,200)
    
    
    ##--create a plot between evaluation values list, and thresholds. ----------###
    
    prelist=[]
    reclist=[]
    f1list=[]
    loan_list=[]
    
    
    
    
    gold_standard_classify,gold_cognate=create_gold_standard_classify(tuple_list_concept_ipa)
    
    
    for t in threshold:
        print ("When the threshold is %s," %t)
        p,r,f,l=eva(t,tup,gold_standard_classify)
        print ("the precision is %s,"%p)
        print ("the recall is %s,"%p)
        print ("the f1 score is %s,"%p)

        
        prelist.append(p)
        reclist.append(r)
        f1list.append(f)
        loan_list.append(l)
   
        
        
    max_f1=max(f1list)
    best_threshold=threshold[f1list.index(max_f1)]
    #best_loan_list=loan_list[f1list.index(max_f1)]
    
    if plot==True:
        plt.plot(list(threshold),reclist,'g-.', list(threshold),prelist,'b', list(threshold),f1list,'r--')         
        plt.show()
        return prelist,reclist,f1list,loan_list,threshold,gold_cognate
    else:
        return max_f1, best_threshold
    
    
    
    



##using cross validation to check the performance. -------------------##



##the input of this function is a tuple list, 
##in which are tuples contains concept,ipa, and distance.
## format is: ('animal', '0.0273098\n', 'ʒanwar', 'hajvɔn'). There are 25k of this tuple. 
## The output is a train set and a test set.  
              
              
              
def cross_validate_output_train_test_set(tuple_list,k_fold):    
    concepts=[]
    
    for ttt in tuple_list:
        if ttt[0] not in concepts:
            concepts.append(ttt[0])
            
    tuple_list_by_concept=[]
    for c in concepts:
        tem=[]
        for ttt in tuple_list:
            if ttt[0]==c:
                tem.append(ttt)            
        tuple_list_by_concept.append(tem)
    
    
    kf=KFold(n_splits=k_fold,shuffle=True)
    train_set=[]
    test_set=[]
    
    for train,test in kf.split(tuple_list_by_concept):
        train_tuple=[]
        test_tuple=[]
        for t in train:
            for tup in tuple_list_by_concept[t]:
                train_tuple.append(tup)
                    
        for t in test:
            for tup in tuple_list_by_concept[t]:
                test_tuple.append(tup)
        
        
        train_set.append(train_tuple)
        test_set.append(test_tuple)
        
    return train_set,test_set





##divide the tuples into train set and test set. 
#train_tuples,test_tuples=cross_validate_output_train_test_set(ipa_distance_tuple_list,10)



## Explore the data we have. 

s_precision,s_recall,s_f1,loan,s_threshold_list,gold_cognate=best_threshold(ipa_distance_tuple_list,plot=True,cv=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





