#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:08:24 2017

@author: kevin
"""

import urllib, json
import pymongo
'''Get the records from fda'''
def getRecords(num_records = 100):
    #num_records = 4587015
    records = []
    i = 0
    while i in range(0,num_records):#go through all records
        i+=100
        url = "https://api.fda.gov/drug/event.json?search=receivedate:[20120101+TO+20161231]&limit=100" + "&skip=" + str(i)
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        results = data["results"]
        records.extend(results)
    '''
    #save data
    with open('drug_adverse_events.json', 'w') as outfile:
        json.dump(records, outfile)
        
    for record in records:
        print record["transmissiondate"]
    '''
    return records

'''Start mongodb, return the fda_records collection for querying'''
def startmongodb():
    from pymongo import MongoClient
    client = MongoClient()
    db = client.cbb750_final_database
    collection = db['fda_records']
    return collection
'''
This gives us the required key/vals.
'''
def getNessessaryKeyValues(record):
    fields = ['safetyreportid',
            'receivedate',
            'serious',
            'seriousnessdeath',
            'seriousnessdisabling',
            'seriousnesshospitalization',
            'seriousnesslifethreatening',
            'seriousnessother',
            'transmissiondate',
            'duplicate',
            'companynumb',
            'occurcountry',
            'primarysourcecountry',
            'primarysource',
            'primarysource.qualification',
            'primarysource.reportercountry',
            'reportduplicate',
            'reportduplicate.duplicatesource',
            'reportduplicate.duplicatenumb',
            "patient.patientonsetage",
            "patient.patientonsetageunit",
            "patient.patientsex",
            "patient.patientweight",
            "patient.patientdeath",
            "patient.patientdeath.patientdeathdate",
            "patient.drug.actiondrug",
            "patient.drug.drugadditional",
            "patient.drug.drugcumulativedosagenumb",
            "patient.drug.drugcumulativedosageunit",
            "patient.drug.drugdosageform",
            "patient.drug.drugintervaldosagedefinition",
            "patient.drug.drugrecurreadministration",
            "patient.drug.drugseparatedosagenumb",
            "patient.drug.drugadministrationroute",
            "patient.drug.drugcharacterization",
            "patient.drug.drugdosagetext",
            'patient.drug.drugenddate',
            'patient.drug.drugindication',
            'patient.drug.drugstartdate',
            'patient.drug.drugtreatmentduration',
            'patient.drug.drugtreatmentdurationunit',
            'patient.drug.medicinalproduct',
            'patient.drug.brand_name',
            'patient.drug.generic_name',
            'patient.drug.manufacturer_name',
            'patient.drug.nui',
            'patient.drug.package_ndc',
            'patient.drug.pharm_class_cs',
            'patient.drug.pharm_class_epc',
            'patient.drug.pharm_class_pe',
            'patient.drug.pharm_class_moa',
            'patient.drug.product_ndc',
            'patient.drug.rxcui',
            'patient.drug.substance_name',
            'patient.reaction.reactionmeddrapt',
            'patient.reaction.reactionmeddraversionpt',
            'patient.reaction.reactionoutcome'
            ]
    subrecord = {}
    for f in fields:
        subrecord.update(get_dotted(record, f))
    return subrecord


'''
This gives us the keys using the dot notation. It also gives us an empty 
string if we dont find the key. It also checks if the next value is inside of a list.
Some work needs to be done on this. I am not keeping the structure. We dont need
the structure to stay the same

TODO:
    We need to make sure that we can go in through an arbitrary number of lists.
    It works for an arbitrary number of dicts.
'''

def get_dotted(d, keys):
    if "." in keys:
        key, rest = keys.split(".", 1)
        if type(d) == list:#make sure its not a list
            d = d[0]
        if type(d[key]) != list and type(d[key]) != dict:#make sure its either a dict or list, otherwise return empty
            if keys not in d:
                d[keys] = ""#some empty value
            return {rest: d[key]}
        return get_dotted(d[key], rest)
    else:
        if type(d) == list:#make sure its not a list
            d = d[0]
        if keys not in d:
            d[keys] = ""#some empty value
        
        subDic = {keys: d[keys]}
        return subDic

def convert_to_float(s):
    try:
        return float(s)
    except ValueError:
        return s

if __name__ == '__main__':
    #main method
    records = getRecords()#get records
    
    subrecords = []#Get all the records and pull out the stuff we need
    for record in records:
        subrecords.append(getNessessaryKeyValues(record))
        
    collection = startmongodb()#get collection
    collection.count()
    collection.insert_many(subrecords)
    print(collection.find_one())
    
    
    
