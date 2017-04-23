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
            #'primarysource',
            'qualification',
            'reportercountry',
            #'reportduplicate',
            'duplicatesource',
            'duplicatenumb',
            "patientonsetage",
            "patientonsetageunit",
            "patientsex",
            #"patient.patientweight",#interseting, this data may be useful but is not recorded much
            "patientdeath",
            "patientdeathdate",
            "actiondrug",
            #"patient.drug.drugadditional",
            #"patient.drug.drugcumulativedosagenumb",
            #"patient.drug.drugcumulativedosageunit",
            "drugdosageform",
            #"patient.drug.drugintervaldosagedefinition",
            #"patient.drug.drugrecurreadministration",
            #"patient.drug.drugseparatedosagenumb",
            "drugadministrationroute",
            "drugcharacterization",
            "drugdosagetext",
            'drugenddate',#sometimes patients do not have an end date? maybe replace with some value like -1
            'drugindication',
            'drugstartdate',
            #'patient.drug.drugtreatmentduration',
            #'patient.drug.drugtreatmentdurationunit',
            'medicinalproduct',
            'brand_name',
            'generic_name',
            'manufacturer_name',
            'nui',
            'package_ndc',
            'pharm_class_cs',
            'pharm_class_epc',
            'pharm_class_pe',
            'pharm_class_moa',
            'product_ndc',
            'rxcui',
            'substance_name',
            'reactionmeddrapt',
            'reactionmeddraversionpt',
            'reactionoutcome'
            ]
    subrecord = {}
    for f in fields:
        extracted_record = get_recursively(record, f)
        #we get the key and try to convert to float, otherwise return the same val
        s_record = {}
        if not extracted_record:
            s_record[f] = None
        elif(len(extracted_record) < 2):#if the result is just 1 value, then return that value not a list
            s_record[f] = extracted_record[0]
        else:
            s_record[f] = extracted_record
        subrecord.update(s_record)
    return subrecord



def get_recursively(search_dict, field):
    """Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.iteritems():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)
    #we want to return a list of values, and we do not want to try and iterate over some none type
    
    new_field_found = []
    if type(fields_found) == list:
        for sublist in fields_found:
            if type(sublist) == list:
                for val in sublist:
                    new_field_found.append(convert_to_float(val))#also convert to float
            else:
                new_field_found.append(convert_to_float(sublist))#also convert to float
    return list(set(new_field_found))

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
                d[keys] = None#some empty value
            return {rest: d[key]}
        return get_dotted(d[key], rest)
    else:
        if type(d) == list:#make sure its not a list
            d = d[0]
        if keys not in d:
            d[keys] = None#some empty value
        
        subDic = {keys: d[keys]}
        return subDic

def convert_to_float(s):
    try:
        return float(s)
    except ValueError:
        return s
    except TypeError:
        return s

def get_y_vals(df, y_label):
    #df.loc[pandas.isnull(df[y_label])] = ""#missing data
    #df.loc[df[y_label].str.contains("", na=False)] = 0
    y = df[y_label]
    del df[y_label]
    return (df, y)#return the new dataframe and y array

'''
Returns a trained naive_bays model
'''

def naive_bays(x_train, y_train, bagging = False, boosting = False):
    from sklearn.naive_bayes import GaussianNB as NB
    nb = NB()
    if bagging == True and boosting == True:
        raise ValueError("Cant have bagging and boosting enabled at the same time")
    if bagging == True:#if bagging
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier (nb, max_samples=.5, max_features=.5)
    elif boosting == True:
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(nb,
                         algorithm="SAMME",
                         n_estimators=300)
    else:#just regular logistic regression
        model = nb
    model.fit(x_train, y_train)
    return model
'''
Returns a trained logistic regression model
'''
def logistic_regression(x_train, y_train, bagging = False, boosting = False):
    from sklearn import linear_model
    lr = linear_model.LogisticRegression()
    if bagging == True and boosting == True:
        raise ValueError("Cant have bagging and boosting enabled at the same time")
    if bagging == True:#if bagging
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier (lr, max_samples=.5, max_features=.5)
    elif boosting == True:
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(lr,
                         algorithm="SAMME",
                         n_estimators=300)
    else:#just regular logistic regression
        model = lr
    model.fit(x_train, y_train)
    return model

def random_forest(x_train, y_train, estimators = 100, maxDepth = None, randomState = 10, maxFeatures = 'auto'):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=estimators, max_depth=maxDepth,
                           random_state=randomState, max_features=maxFeatures)
    model.fit(x_train, y_train)
    return model



#Auto encodes any dataframe column of type category or object.
def dummyEncode(df):
    from sklearn.preprocessing import LabelEncoder
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

if __name__ == '__main__':
    #main method
    records = getRecords(num_records=1000)#get records
    
    subrecords = []#Get all the records and pull out the stuff we need
    for record in records:
        subrecords.append(getNessessaryKeyValues(record))
    
    import pandas
    #x_y_dataframe = pandas.DataFrame(subrecords,index=[range(0,len(subrecords))])#put into a dataframe
    x_y_dataframe = pandas.DataFrame(subrecords)
    x_y_dataframe2 = x_y_dataframe.dropna(axis=0)
    #get labels
    (x,y) = get_y_vals(x_y_dataframe, 'serious')
    #df = pandas.DataFrame([{'drug': ['drugA','drugB'], 'patient': 'john'}, {'drug': ['drugC','drugD'], 'patient': 'angel'}])
    
    res = []
    categorical_labels = [
 'brand_name',
 'companynumb',
 'drugadministrationroute',
 'drugcharacterization',
 'drugdosageform',
 'drugdosagetext',
 #'drugenddate',
 'drugindication',
 'drugstartdate',
 #'duplicate',
 'duplicatenumb',
 'duplicatesource',
 'generic_name',
 'manufacturer_name',
 'medicinalproduct',
 #'nui',
 'occurcountry',
 'package_ndc',
 'patientdeath',
 'patientdeathdate',
 #'patientonsetage',
 'patientonsetageunit',
 'patientsex',
 'pharm_class_cs',
 'pharm_class_epc',
 'pharm_class_moa',
 'pharm_class_pe',
 'primarysourcecountry',
 'product_ndc',
 #'qualification',
 'reactionmeddrapt',
 'reactionmeddraversionpt',
 'reactionoutcome',
 #'receivedate',
 'reportercountry',
 'rxcui',
 #'safetyreportid',
 'seriousnessdeath',
 'seriousnessdisabling',
 'seriousnesshospitalization',
 'seriousnesslifethreatening',
 'seriousnessother',
 'substance_name',
 #'transmissiondate'
 ]
    for col in categorical_labels:
        v = x[col].astype(str).str.strip('[]').str.get_dummies(', ')
        res.append(v)
    x = pandas.concat(res, axis=1)
    y = pandas.to_numeric(y.astype(str).str.strip('[]'))
    
#    y[y.isnull()] = 0#we do this to convert nans, if its empty, we set it to 0
    #y[0:20] = 1#just a test
#    import numpy as np
#    x[np.isfinite(x['drugintervaldosagedefinition'])]
    #x_y_dataframe.dropna(how='any')#here we can drop a row if there is any na
    #split the data:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    #do a logistic regression
    model = logistic_regression(X_train, list(y_train), boosting=False)
    y_hat = model.predict(X_test)#predict
    y_test = y_test.reset_index(drop=True)    
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_hat)
    
    
    #do a random forest
    model = random_forest(X_train, list(y_train))
    y_hat = model.predict(X_test)#predict
    y_test = y_test.reset_index(drop=True)    
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_hat)
    
    #do a naive bays
    model = naive_bays(X_train, list(y_train), bagging=True)
    y_hat = model.predict(X_test)#predict
    y_test = y_test.reset_index(drop=True)    
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_hat)    
    
    
    collection = startmongodb()#get collection
    collection.count()
    collection.insert_many(subrecords)
    print(collection.find_one())
    
    
    
