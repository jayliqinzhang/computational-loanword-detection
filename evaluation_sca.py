import os
from collections import defaultdict

import itertools

import numpy as np

from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


folder='cm1_result/'

## First, we delete the result files without any value. 
## Then we put the file names in a list. 
result_name_list=[]
for file in os.listdir(folder):
        if sum(1 for line in open(folder+file,encoding='utf-8'))<2:
            os.remove(folder+file)
        else:
            result_name_list.append(file)
        
            
#            

groups=defaultdict(list)
for result in result_name_list:
    groups[result.split('_')[1]].append(result)
group_result=groups.values()


## Make a dictionary containing the distance values, giving the file names. 
## The split one format is split_concept_distance_dict[data_animal_0.dat]=[[num...],[num...],[num...]]
split_concept_distance_dict={}
for file in result_name_list:
    with open(folder+file,encoding='utf-8') as f:  
        content=f.read().split('\n')[1:]
        tem=[]
        for c in content:
            try:
                tem.append(float(c.split(',')[2]))
            except IndexError:
                pass
        split_concept_distance_dict[file]=tem

## The combine one format is combine_concept_distance_dict['animal']=[num...]
combine_concept_distance_dict={}
for k,v in groups.items():
    tem=[]
    for i in range(len(v)):
        tem.append(split_concept_distance_dict[v[i]]) 
    if '.' in k:
        combine_concept_distance_dict[k.split('.')[0]]=list(itertools.chain.from_iterable(tem))
    else:
        combine_concept_distance_dict[k]=list(itertools.chain.from_iterable(tem))


         
###-------------------------------------------------------------------------------------------------------##
## Pair the file names, like [animal_tu_0.dat,animal_ir_0]
paired_split_concept_flie_list=[]
for file in os.listdir('data_fil_files_splitlong/'):
    with open ('data_fil_files_splitlong/'+file,encoding='utf-8') as f:
        tem=[]
        for line in f:
            tem.append(line)
    paired_split_concept_flie_list.append(tem)


## Make a dictionary, key is the split concept file names, value is a list of xsampa.  
## format is split_concept_xsampa_dict['animal_tu_0.dat']=['1 Zanwar', ....]
split_concept_xsampa_dict={}
for file in os.listdir('within_cross_data_duplicate_splitlong/'):
    with open ('within_cross_data_duplicate_splitlong/'+file,encoding='utf-8') as f:                  
        tem=[]
        for line in f:
            tem.append(line)
    split_concept_xsampa_dict[file]=tem


##　Now the values in the pair list is replaced by the corresponding xsampa. 
paired_tu_ir_xsampa_list=[]

for pair in paired_split_concept_flie_list:
    tem=[]
    for i in range(len(pair)):
        try:
            if i==0:
                tem.append(split_concept_xsampa_dict[pair[i][:-1]])
            else:
                tem.append(split_concept_xsampa_dict[pair[i]])
        except KeyError:
            pass
    paired_tu_ir_xsampa_list.append(tem)

        
## Now pair every single xsampa.
## The format is: concept_xsampa_dict['animal_tu_0.dat']=[('1 Zanwar\n','1 hajvOn\n'), ....]
## Although the key is with 'tu' inside, it is crossing tu and ir. 
split_concept_xsampa_dict={}
for i in range(len(paired_tu_ir_xsampa_list)):
    try:
        tem=[]
        for j in range(len(paired_tu_ir_xsampa_list[i][0])):  
            tem.append((paired_tu_ir_xsampa_list[i][0][j],paired_tu_ir_xsampa_list[i][1][j]))
            
        if len(paired_split_concept_flie_list[i][0][:-1].split('_'))==3:           
            split_concept_xsampa_dict['data_'+paired_split_concept_flie_list[i][0][:-1].split('_')[0]+'_'+
                                        paired_split_concept_flie_list[i][0][:-1].split('_')[2]]=tem
                                
        else:
            split_concept_xsampa_dict['data_'+paired_split_concept_flie_list[i][0][:-1].split('_')[0]+'.dat']=tem
                                

            
    except IndexError:
        pass



    
combine_concept_xsampa_dict={}
for k,v in groups.items():
    tem=[]
    for i in range(len(v)):
        try:
            tem.append(split_concept_xsampa_dict[v[i]]) 
        except KeyError:
            try:
                tem.append(split_concept_xsampa_dict['data_to-'+v[i].split('_')[1]+'_'+v[i].split('_')[2]])
            except IndexError:
                tem.append(split_concept_xsampa_dict['data_to-'+v[i].split('_')[1]])

    if '.' in k:
        combine_concept_xsampa_dict[k.split('.')[0]]=list(itertools.chain.from_iterable(tem))
    else:
        combine_concept_xsampa_dict[k]=list(itertools.chain.from_iterable(tem))
   


## We need import ip_xsampa_words_tuple_list first to run this part. 
## It is a list containing pairs of corresponding ipa and xsampa. 
## It is used in this function, to transform xsampa to ipa. 
## The tuple_list arguement is in fact the ip_xsampa_words_tuple_list in later used. 
def from_x_to_ipa(xsampa,tuple_list):
       
        for pair in tuple_list:
            if '\n' in xsampa: 
                if pair[1]==xsampa[:-1]:
                    return pair[0]
            else:
                if pair[1]==xsampa:
                    return pair[0]






## Running this part takes times, so the ipa_distance_tuple_list is saved,
## and this part is comment out. 


## Need to make a tuple like this: (concept, distance, ipa,ipa)
## ('animal', '0.0273098\n', 'ʒanwar', 'hajvɔn')
## This tuple list is used in the later evaluation section directly.

ipa_distance_tuple_list=[]  

for k,v in combine_concept_distance_dict.items():
        
        for i in range(len(combine_concept_xsampa_dict[k])):
            try:
                x1=from_x_to_ipa(combine_concept_xsampa_dict[k][i][0],ipa_xsampa_words_tuple_list)
                x2=from_x_to_ipa(combine_concept_xsampa_dict[k][i][1],ipa_xsampa_words_tuple_list)
                    
                
                ipa_distance_tuple_list.append((k,combine_concept_distance_dict[k][i],x1.split()[1],x2.split()[1]))
            except IndexError:
                #print(combine_concept_xsampa_dict[k][i])
                pass

dis_list=[]
for i in ipa_distance_tuple_list:
    dis_list.append(i[1])
                  

####------------------------------------------------------------------------#########

## Preparation for evaluation. First, use the imported cognate_list data. 
## output the cognate_not_distionary. 
 


### a all_ipa_tur_ira_cross list is needed to marked the 0 and 1. 
### The format is (concept, ipa, ipa).
all_ipa_tur_ira_cross=[]

for tup in ipa_distance_tuple_list:       
    all_ipa_tur_ira_cross.append((tup[0],tup[2],tup[3]))
                  




##extract gold standard------------------------###
##cognate list contains the cognates with letters.  
cognate_not_distionary=[]
for k,v in cognate_list.items():
    for loans in v:
        cognate_not_distionary.append((k,loans[0][0],loans[1][0]))
    



##make note. assgin '1' if a pair is loan, assign '0' if a pair is not loan. 
def create_gold_standard_classify(tup_concept_tur_ira_in_ipa):    
    gold_standard_classify=[]
    for aaa in tup_concept_tur_ira_in_ipa:
        if aaa in cognate_not_distionary:
            gold_standard_classify.append(1)
        else:
            gold_standard_classify.append(0)
    return gold_standard_classify




###########_--For evaluation--------------------------------------

##---The input of the function is a list of thresholds. 
##---For one threshold, there is one accurach, precision, recall and f1 value. 
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
            
            
#        
#    for k,v in no_duplicate_dic_ipa_and_dis.items():
#        #print(v[1])
#        if float(v[1])<thres:
#            try:
#                predict_loanwords.append((k.split('_')[3],v[0][0][:-1],v[0][1][:-1]))
#            except IndexError:
#                pass
    
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

def best_threshold(tup, plot=False):
    ##--make a list of distance.----------------------------####
    dis_values_list=[]
    for t in tup:
        dis_values_list.append(float(t[1]))
               
    tuple_list_concept_ipa=tuple_with_dis_to_tuple_no_dis(tup)  ## This is the tuple list used for creating gold starndard.   

    
#    
#    for k,v in dictionary_ipa_and_dis_no_duplicate.items():
#        if k not in needed_to_delete:
#            dis_values_list.append(float(dictionary_ipa_and_dis_no_duplicate[k][1]))
        
    
    ave_distance=average(dis_values_list)    
    ## Looking for the mimimum value other than 0. 
    min_distance=min(x for x in dis_values_list if x>0) 
    
    ## There are 10 potential thresholds in case of cross evaluation.
    ## But to check the whole the dataset for once, it should be much bigger, like 200. 
    threshold=np.linspace(min_distance,ave_distance,200)
    


    
    ##--create a plot between evaluation values list, and thresholds. ----------###
    
    prelist=[]
    reclist=[]
    f1list=[]
    loan_list=[]
    
    
    
    
    gold_standard_classify=create_gold_standard_classify(tuple_list_concept_ipa)
    
    
    for t in threshold:
        p,r,f,l=eva(t,tup,gold_standard_classify)
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
        return prelist,reclist,f1list,loan_list,threshold
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





#divide the tuples into train set and test set. 
#train_tuples,test_tuples=cross_validate_output_train_test_set(ipa_distance_tuple_list,10)
#
#
#cv_pre_list=[]
#cv_rec_list=[]
#cv_f1_list=[]
#cv_predict_result=[]
#cv_gold_from_test_set=[]
#for t in range(len(train_tuples)):
#    
#    b_f1,b_thres=best_threshold(train_tuples[t])
#    
#    
#    
#    gold_from_test=create_gold_standard_classify(tuple_with_dis_to_tuple_no_dis(test_tuples[t]))
#    
#    pre,rec,f1,loan=eva(b_thres,test_tuples[t],gold_from_test)
#      
#    
#    print(pre,'   ',rec,'   ',f1)
#    
#    cv_pre_list.append(pre)
#    cv_rec_list.append(rec)
#    cv_f1_list.append(f1)
#    cv_predict_result.append(loan)
#    cv_gold_from_test_set.append(gold_from_test)
#    
#



## Explore the data we have. 

s_precision,s_recall,s_f1,loan,s_threshold_list=best_threshold(ipa_distance_tuple_list,plot=True)

#print (max(s_f1))

#print (s_threshold_list[s_f1.index(max(s_f1))])                     
                             



                             
                             
                             
                             
                             