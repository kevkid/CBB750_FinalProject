#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:23:50 2017

@author: RajBrarMD
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:22:31 2017

@author: RajBrarMD
"""

import requests
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import operator

# Load in data from json file

#with open('data.json') as json_data:
#    d = json.load(json_data)
#
## Get all rxcui's in data 
#rxnorm = []
#for i in range(0,len(d)-1):
#    try:
#        tmp = d[i]['patient']['drug'][0]['openfda']['rxcui'] # This returns a list  
#        rxnorm.append(tmp)
#    except Exception, e:
#        pass

def rxnorm(rxnorm):
    #rxnorm = list(itertools.chain.from_iterable(rxnorm)) # Flatten into one list
    
    # Before taking unique set of rxcui, get counts of each and put them into a dictionary
    from collections import Counter
    rxcui_counts = Counter(rxnorm)
     
    rxcui_unique = list(set(rxnorm)) # Get unique rxcui values
    
    # Build dictionary with rxcui as key, and values of rxcui_count, drug name, and drug class
    count = 1
    len_rxcui_unique = len(rxcui_unique)
    rxcui_dict = {}
    for rxcui in rxcui_unique:
        print 'rxcui request: {}/{}'.format(count, len_rxcui_unique)
        count += 1
        searchURL = 'https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui=' + str(rxcui) + '&relaSource=FDASPL'
        req = requests.get(searchURL)
        req = req.json()
         
        try:
            rxcui_dict[rxcui] = {
                             'drug_name': (req['rxclassDrugInfoList']['rxclassDrugInfo'][1]['minConcept']['name'],rxcui_counts[rxcui]),
                             'drug_class': (req['rxclassDrugInfoList']['rxclassDrugInfo'][1]['rxclassMinConceptItem']['className'],rxcui_counts[rxcui])}
        except Exception, e:
            pass
        
    # Extract drugs and classes with their associated frequencies (in the form of tuples)
    drugs = list(rxcui_dict[key]['drug_name'] for key in rxcui_dict.keys())
    classes = list(rxcui_dict[key]['drug_class'] for key in rxcui_dict.keys())
    
    # Aggregate 2nd value in tuple (count) based on 1st value (drugs and classes)
    l = [(uk,sum([vv for kk,vv in drugs if kk==uk])) for uk in set([k for k,v in drugs])]
    l2 = [(uk,sum([vv for kk,vv in classes if kk==uk])) for uk in set([k for k,v in classes])]
        
    
    # Get total number of occurences of drug, will be same as occurences of class
    total = sum(rxcui_dict[key]['drug_name'][1] for key in rxcui_dict.keys())
    
    # Sort by 2nd value of tuple (count) and take top 10
    l = sorted(l, key=lambda x: x[1])[-10:]
     
    l2 = sorted(l2, key=lambda x: x[1])[-10:]
     
    
    # Plot top 10 drugs and classes (by % of total)
    fig, ax = plt.subplots()
    ax.barh(range(len(l)), [(t[1]/float(total)) * 100 for t in l]  , align="center" )
    ax.set_yticks(range(len(l)))
    ax.set_yticklabels([t[0] for t in l])
    ax.set_xlabel('% of Total Drugs')
    ax.set_title('Top 10 Drugs by Appearance in Event Reports')
    ax.grid(True)
    
    plt.show()
    
    fig, ax = plt.subplots()
    ax.barh(range(len(l2)), [(t[1]/float(total)) * 100 for t in l2]  , align="center" )
    ax.set_yticks(range(len(l2)))
    ax.set_yticklabels([t[0] for t in l2])
    ax.set_xlabel('% of Total Drugs')
    ax.set_title('Top 10 Drug Classes')
    ax.grid(True)
    
    plt.show()
     
    return rxcui_dict