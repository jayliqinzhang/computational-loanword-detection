from lingpy import *

## the orignal concepts-turkic-indo_iranian_ipa data is reformated in a csv file.
## the csv file is read to a dictionary. 

def sca_distance(csvfile):

    ## input your csv file and handle it via the lingpy function to handle csv file. 
    ## The seperator and the header can be changed according to the format of your csv file. 
    data_dictionary=read.csv.csv2dict(csvfile,sep=';',header=True) 
    
    
    # function that delete the duplicate elements in a list. 
    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    
    
    
    ipa_pair_dict={}
    for k,v in data_dictionary.items():
        tu=f7(list(filter(None,v[0][:-2].split(' / '))))
        ir=f7(list(filter(None,v[1][:-2].split(' / '))))
        for t in tu:
            if '/' in t:
                tu.remove(t)
        for i in ir:
            if '/' in i:
                ir.remove(i)
               
        for i in range(len(tu)):
            for j in range(len(ir)):
                ipa_pair_dict[str(k)+'_'+str(i)+'_'+str(j)]=(ipa2tokens(tu[i]),ipa2tokens(ir[j]))
        

    
    ## http://lingpy.org/reference/lingpy.align.html#lingpy.align.pairwise.Pairwise
    
    ## Calculating the distance between two pronunciations using SCA distance. 
    ipa_distance_tuple_list=[]
    
    
    for k,v in ipa_pair_dict.items():
        
    
        align_pair_ipa=align.pairwise.Pairwise(v[0],v[1])
        
        align_pair_ipa.align(distance=True,method='sca')
    
        (ipa1,ipa2,distance)=align_pair_ipa.alignments[0]
            
        ipa_distance_tuple_list.append((k.split('_')[0],distance,''.join(v[0]),''.join(v[1])))
                
    
    ## the function outputs in a format as following: 
    ## (concept, distance value, ipa in turkic, ipa in indo-inranian). for instance, ('one', 0.84, 'bɪr', 'jak') 
    
    return ipa_distance_tuple_list



result=sca_distance('data/ipa_data.csv')



