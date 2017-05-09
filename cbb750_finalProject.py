#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:08:24 2017

@author: kevin
"""

import urllib, json
import pymongo
import os
os.chdir("Downloads/CBB750_FinalProject/") 


'''Get the records from fda'''
def getRecords(num_records = 100, start_date='20120101', end_date='20161231'):
    #num_records = 4587015
    records = []
    i = 0
    while i in range(0,num_records):#go through all records
        i+=100
        #i = i%5000#limit for skip, as per FDA api
        url = "https://api.fda.gov/drug/event.json?api_key=QIYs3xaRHyjYEkcprTinxbS6QXABp4cu55ZcR2iV&search=receivedate:[" + str(start_date) + "+TO+" + str(end_date) + "]&limit=100" + "&skip=" + str(i)
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        print i
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
            #'seriousnessdeath',
            #'seriousnessdisabling',
            #'seriousnesshospitalization',
            #'seriousnesslifethreatening',
            #'seriousnessother',
            'transmissiondate',
            'duplicate',
            #'companynumb',
            'occurcountry',
            'primarysourcecountry',
            #'primarysource',
            'qualification',
            'reportercountry',
            #'reportduplicate',
            #'duplicatesource',
            #'duplicatenumb',
            #"patientonsetage",
            #"patientonsetageunit",
            "patientsex",
            #"patient.patientweight",#interseting, this data may be useful but is not recorded much
            #"patientdeath",
            #"patientdeathdate",
            "actiondrug",
            #"patient.drug.drugadditional",
            #"patient.drug.drugcumulativedosagenumb",
            #"patient.drug.drugcumulativedosageunit",
            #"drugdosageform",
            #"patient.drug.drugintervaldosagedefinition",
            #"patient.drug.drugrecurreadministration",
            #"patient.drug.drugseparatedosagenumb",
            #"drugadministrationroute",
            "drugcharacterization",
            #"drugdosagetext",
            #'drugenddate',#sometimes patients do not have an end date? maybe replace with some value like -1
            #'drugindication',
            #'drugstartdate',
            #'patient.drug.drugtreatmentduration',
            #'patient.drug.drugtreatmentdurationunit',
            #'medicinalproduct',
            #'brand_name',
            'generic_name',
            #'manufacturer_name',
            #'nui',
            #'package_ndc',
            #'pharm_class_cs',
            #'pharm_class_epc',
            #'pharm_class_pe',
            #'pharm_class_moa',
            #'product_ndc',
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

'''
split arrays into evenly sized chunks
'''
def chunks(array, n):
    """Yield successive n-sized chunks from array."""
    for i in xrange(0, len(array), n):
        yield array[i:i + n]
        
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

def one_hot_encode(categorical_labels):
    import gc
    res = []
    tmp = None
    count = 1
    cat_lab_len = len(categorical_labels)
    for col in categorical_labels:
        print "Label: " + str(count) + '/' + str(cat_lab_len)
        print "Currently on label: " + str(col)
        count+=1
        v = x[col].astype(str).str.strip('[]').str.get_dummies(', ')#cant set a prefix
        if len(res) == 2:
            tmp = pandas.concat(res, axis=1)
            del res
            res = []
            res.append(tmp)
            del tmp
            tmp = None
            gc.collect()
        else:
            res.append(v)
    result = pandas.concat(res, axis=1)
    return result

def getJsonList(directory = "/home/kevin/Downloads/CBB750_FinalProject/mnt"):
    import os, glob
    import json
    os.chdir(directory)
    #take each json file read each list element, and write it to the system
    data = []
    for f in glob.glob("*.json"):
        data.append(f)
    return data

def get_top_categorical(categorical_labels, N = 50):
    count = 1
    cat_lab_len = len(categorical_labels)
    for label in categorical_labels: 
        print 'Labels: ' + str(count) + '/' + str(cat_lab_len)
        count +=1
        from collections import defaultdict, Counter
        labelDict = defaultdict(int)
        for element in x_y_dataframe[label]:
            if type(element) == list:
                for el in element:#for every value in the list
                    labelDict[el] += 1
            else:
                labelDict[element] += 1
        
        labelDict = dict(Counter(labelDict).most_common(N))
        labelDictKeys = labelDict.keys()
        labelDictKeysSet = set(labelDictKeys)
        for idx in range(0, len(x_y_dataframe[label])):
            val = x_y_dataframe[label][idx]
            if type(val) == list:#do an intersection
                res = list(set(val) & labelDictKeysSet)
                if not res:#if list is empty
                    x_y_dataframe.set_value(idx,label, None)
                elif(len(res) == 1):
                    x_y_dataframe.set_value(idx,label, res[0])
                else:
                    x_y_dataframe.set_value(idx,label, res)
            else:
                if val not in labelDictKeys:
                    x_y_dataframe.set_value(idx,label, None)
def fix_date(column):
    for col in column:
        x_y_dataframe[col] = pandas.to_datetime(x_y_dataframe[col], errors='coerce')
if __name__ == '__main__':
    #main method
#    records = getRecords(num_records=5000)#get records
#    
#    #save to disk
#    with open('data.json', 'w') as outfile:
#        json.dump(records, outfile)
#    #read from disk
#    with open('data.json') as json_data:
#        records = json.load(json_data)
    
    categorical_labels = [
                     'actiondrug',
                     #'brand_name',
                     #'companynumb',
                     #'drugadministrationroute',
                     'drugcharacterization',
                     #'drugdosageform',
                     #'drugdosagetext',
                     #'drugenddate',
                     #'drugindication',
                     #'drugstartdate',
                     #'duplicate',
                     #'duplicatenumb',
                     #'duplicatesource',
                     'generic_name',
                     #'manufacturer_name',
                     'medicinalproduct',
                     #'nui',#has lists
                     'occurcountry',
                     #'package_ndc',
                     #'patientdeath',
                     #'patientdeathdate',
                     #'patientonsetage',
                     #'patientonsetageunit',
                     'patientsex',
                     #'pharm_class_cs',
                     #'pharm_class_epc',
                     #'pharm_class_moa',
                     #'pharm_class_pe',
                     'primarysourcecountry',
                     #'product_ndc',
                     #'qualification',
                     'reactionmeddrapt',
                     'reactionmeddraversionpt',
                     'reactionoutcome',
                     #'receivedate',
                     'reportercountry',
                     'rxcui',#lets pretend its not a categorical
                     #'safetyreportid',
                     #'seriousnessdeath',
                     #'seriousnessdisabling',
                     #'seriousnesshospitalization',
                     #'seriousnesslifethreatening',
                     #'seriousnessother',
                     'substance_name',
                     #'transmissiondate'
                     ]
    '''
    We can read in each json file individually and extract the neccessary
    elements we need thus making the file smaller and easier to read.
    Here we read it in and save the data
    '''
    data = getJsonList("/media/kevin/Anime/drug-event_2013")#get the list of json files
    #os.chdir("/home/kevin/Downloads/CBB750_FinalProject/") 
    subrecords_loc = '/home/kevin/Downloads/CBB750_FinalProject/subrecords_.json'
    num_files = len(data)
    count  = 0
    subrecords = []#Get all the records and pull out the stuff we need

    
    for element in data:#get all of the files from some server
        count +=1
        print "Files Progress: " + str(count) + '/' + str(num_files)
        
        with open(element) as json_data:#for each file load it and get its results
            json_file = json.load(json_data)
            
        records = json_file['results']#just get results
        del json_file#save memory
        num_rec = len(records)
        rec_count = 0
        rec = []
        
        for record in records:#for all of the records in the file prune the record and get what we need
            #rec_count += 1
            #print "Records per file Progress: " + str(rec_count) + '/' + str(num_rec)
            rec = (getNessessaryKeyValues(record))#pruning
            
            
            subrecords.append(rec)#append the record to a list
    print "Writing to json file"
    with open('/home/kevin/Downloads/CBB750_FinalProject/subrecords_.json', "a") as out_file:
        json.dump(subrecords, out_file)
                    
    data = subrecords
    del records#save memory
    del record
    del subrecords
    
    os.chdir("/home/kevin/Downloads/CBB750_FinalProject/")
    '''
    Read in the subarray data
    '''
    #read from disk
    with open('/home/kevin/Downloads/CBB750_FinalProject/subrecords_.json', 'r') as data_file:
        data = json.load(data_file)
        
        
    import pandas
    import numpy as np
    x_y_dataframe = pandas.DataFrame(data)
    #x_y_dataframe.dropna(how='any')#here we can drop a row if there is any na
    del data
    #remove columns that have the most nas
    fix_date(['receivedate'])
    get_top_categorical(['generic_name', 'medicinalproduct', 'substance_name'], 500)
    x_y_dataframe = x_y_dataframe.dropna(axis=0)
    x_y_dataframe = x_y_dataframe.reset_index(drop=True)
    x_y_dataframe = x_y_dataframe.iloc[0:100000]
#    del x_y_dataframe
#    df_arr = np.array_split(x_y_dataframe, 3)
    

#    
#

    

    
    
    
    #check columns for most nas
#    cols = list (x_y_dataframe)
#    d = {}
#    for col in cols:
#        #d[col] = 
#        print x.loc[x[col].str.contains("12.5 MG EACH ANTIEMETIC") ]
        #print col + ": " + str(x_y_dataframe[col].str.contains("12.5 MG EACH ANTIEMETIC"))
    
    #get labels
    (x,y) = get_y_vals(x_y_dataframe, 'serious')
    del x_y_dataframe
    
    #df = pandas.DataFrame([{'drug': ['drugA','drugB'], 'patient': 'john'}, {'drug': ['drugC','drugD'], 'patient': 'angel'}])
    
    #save data
    x.to_pickle('x_data.pkl')
    y.to_pickle('y_data.pkl')
    #load data
    import pandas
    x = pandas.read_pickle('x_data.pkl')
    y = pandas.read_pickle('y_data.pkl')
    
    '''
    get difference in labeles we are one-hot encoding and ones we aren't
    We need to preserve the ones we arent one-hot encoding
    '''
    s = set(categorical_labels)
    not_categorical_labels = [x_val for x_val in list(x) if x_val not in s]
    x_non_cat_labels = x[not_categorical_labels]#save non cat to temp var
    x = one_hot_encode(categorical_labels)
    
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer()
    X = vect.fit_transform(x_y_dataframe.medicinalproduct.map(lambda x: ' '.join(x) if isinstance(x, list) else x))
    r = pandas.SparseDataFrame(X, columns=vect.get_feature_names(), index=x_y_dataframe.index, default_fill_value=0)
    r = pandas.DataFrame(X.A, columns=vect.get_feature_names(), index=x.index)
    m = pandas.concat([m,r], axis=1)
    
    
    
    
    
    num_processors = 1
    n_size = int(round(len(categorical_labels)/num_processors))
    import time
    import multiprocessing
    os.system("taskset -p 0xff %d" % os.getpid())
    start_time = time.time()
    p = multiprocessing.Pool(num_processors)
    res = (p.map(one_hot_encode, list(chunks(categorical_labels, n_size))))
    print (time.time()-start_time)
    p.terminate()#kill the workers
    res.append(x_non_cat_labels)
    del x_non_cat_labels
    x = pandas.concat(res, axis=1)
    y = pandas.to_numeric(y.astype(str).str.strip('[]'))
    x = x.dropna(how='any')
    y = y[x.index]
#    y[y.isnull()] = 0#we do this to convert nans, if its empty, we set it to 0
    #y[0:20] = 1#just a test
#    import numpy as np
#    x[np.isfinite(x['drugintervaldosagedefinition'])]
    #x_y_dataframe.dropna(how='any')#here we can drop a row if there is any na
    #split the data:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    del x, y
    #do a logistic regression
    model = logistic_regression(X_train, list(y_train), bagging=True)
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
    
    
    
