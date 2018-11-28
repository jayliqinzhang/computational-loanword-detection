import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import csv


distance=[]
with open('data/sca_distance.csv',newline='\n') as f:
    csvreader=csv.reader(f,delimiter=',')
    for row in csvreader:          
        distance.append(tuple(row))
    
cognate_list=[]
with open('data/cognate.csv',newline='\n') as f:
    csvreader=csv.reader(f,delimiter=',')
    for row in csvreader:          
        cognate_list.append(tuple(row))


all_ipa_tur_ira_cross=[]

for tup in distance:       
    all_ipa_tur_ira_cross.append((tup[0],tup[2],tup[3]))
                  



##make note. assgin '1' if a pair is loan, assign '0' if a pair is not loan. 
def create_gold_standard_classify(tup_concept_tur_ira_in_ipa):    
    gold_standard_classify=[]
    gold_standard_cognate=[]
    for aaa in tup_concept_tur_ira_in_ipa:
        if aaa in cognate_list:
            gold_standard_classify.append(1)
            gold_standard_cognate.append(aaa)
        else:
            gold_standard_classify.append(0)
    return gold_standard_classify,gold_standard_cognate





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

def get_threshold(tup, plot=False):
    ##--make a list of distance.----------------------------####
    dis_values_list=[]
    for t in tup:
        dis_values_list.append(float(t[1]))
               
    tuple_list_concept_ipa=tuple_with_dis_to_tuple_no_dis(tup)  ## This is the tuple list used for creating gold starndard.   

 
    
    ave_distance=average(dis_values_list)    
    ## Looking for the mimimum value other than 0. 
    min_distance=min(x for x in dis_values_list if x>0) 
    
    threshold=np.linspace(min_distance,ave_distance,10)
    
    
    ##--create a plot between evaluation values list, and thresholds. ----------###
    ## It shows how the performance changes with different threshold values. 
    prelist=[]
    reclist=[]
    f1list=[]
    loan_list=[]
    
    gold_standard_classify,gold_cognate=create_gold_standard_classify(tuple_list_concept_ipa)
    
    
    for t in threshold:
        p,r,f,l=eva(t,tup,gold_standard_classify)
        print ("When the threshold is %s," %t)
        print ("the precision is %s,"%p)
        print ("the recall is %s,"%p)
        print ("the f1 score is %s,"%p)
        print ('\n')

        
        prelist.append(p)
        reclist.append(r)
        f1list.append(f)
        loan_list.append(l)
   
        
        
    max_f1=max(f1list)
    best_threshold=threshold[f1list.index(max_f1)]
    print('the threshold value leads to the best performance is %s' %best_threshold)
    print('the f1 score is %s' %max_f1)
    
    best_loan_list=loan_list[f1list.index(max_f1)]

    
    if plot==True:
        plt.plot(list(threshold),reclist,'g-.', list(threshold),prelist,'b', list(threshold),f1list,'r--')         
        plt.show()
    
    return best_threshold,best_loan_list

    

    
best_threshold,loanword_list=get_threshold(distance,plot=True)


    

