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


book=xlrd.open_workbook("mydata.xls")

sheet=book.sheet_by_index(0)


concepts=sheet.row_values(0,start_colx=1)
turkic=sheet.row_values(1,start_colx=1)
iranian=sheet.row_values(2,start_colx=1)

# function that delete the duplicate elements in a list. 
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

    
for i in range(len(turkic)):
    turkic[i]=f7(list(filter(None,turkic[i].split(' / '))))  ##split the file, filter the empty, and delete duplicate.
    iranian[i]=f7(list(filter(None,iranian[i].split(' / '))))


concept_tu_ipa_dict={}
concept_ir_ipa_dict={}
for i in range(len(turkic)):
    for j in range(len(turkic[i])):
        concept_tu_ipa_dict[concepts[i]+'_'+str(j)]=ipa2tokens(turkic[i][j])
    for j in range(len(iranian[i])):
        concept_ir_ipa_dict[concepts[i]+'_'+str(j)]=ipa2tokens(iranian[i][j])


ipa_pair_dict={}   ## the dictionary in which the keys are the concepts and the values are different pairs of pronunciation of the concepts. 
for k_tu,v_tu in concept_tu_ipa_dict.items():
    for k_ir,v_ir in concept_ir_ipa_dict.items():
        if k_tu.split('_')[0]==k_ir.split('_')[0]:
            ipa_pair_dict[k_tu.split('_')[0]+'_'+k_tu.split('_')[1]+'_'+k_ir.split('_')[1]]=(v_tu,v_ir)
           
        


##--------------------finishing handling data, begin to get distance--------------------###

## http://lingpy.org/reference/lingpy.align.html#lingpy.align.pairwise.Pairwise

## Calculating the distance between two pronunciations using SCA distance. 
ipa_distance_tuple_list=[]

tuples_with_alignment=[]

for k,v in ipa_pair_dict.items():
    
    align_pair_ipa=align.pairwise.Pairwise(v[0],v[1])
    align_pair_ipa.align(distance=True,method='sca')

    (ipa1,ipa2,distance)=align_pair_ipa.alignments[0]
    

    ipa_distance_tuple_list.append((k.split('_')[0],distance,''.join(v[0]),''.join(v[1])))
    
    tuples_with_alignment.append((k.split('_')[0],distance,''.join(v[0]),''.join(v[1]),ipa1,ipa2))
    
    
import pickle

with open ('sac_distance.pkl','wb') as f:
    pickle.dump(ipa_distance_tuple_list,f)


    
    
    
    
    








