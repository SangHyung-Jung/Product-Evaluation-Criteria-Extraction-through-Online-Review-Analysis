import os, pickle, time
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

### Before run this py, we need to prepare tagged lad_noun with tagged label ( handwork )

def loadbyvectors(data, ft128):
    load_x = []
    
    for i in range(len(data)):
        try: tmp = ft128.wv[data[i]]
        except:
            print('not in the vector : ', data[i])
            tmp = np.zeros(128, dtype=np.float32)
        load_x.append(tmp)

    load_x = np.asarray(load_x)      

    return load_x

def preprocess():
    ### concat all middle categories tagged files
    file_list=[]
    for file in os.listdir("where\located\tagged_csv"):
        if file.endswith(".csv"): 
            file_list.append(file)

    data_df = pd.read_csv(file_list[0], engine='python')[['noun','label']]        
    for file_name in file_list[1:]:
        df = pd.read_csv(file_name, engine='python')[['noun','label']]
        data_df = pd.concat([data_df, df], axis=0, join='outer')

    data_df.to_csv('tagged_all.csv', index=False, encoding = 'cp949')   

    data_df = pd.read_csv('tagged_all.csv', engine='python')[['noun','label']]
    data_df = data_df.drop_duplicates()

    return data_df

def knn_model_train(data_df):

    with open('ft128.model', 'rb') as f:
        ft128 = pickle.load(f)
    t_data = data_df[data_df['label']==2]
    p_data = data_df[data_df['label']==1]
    n_data = data_df[data_df['label']==0]
    
    t_train, t_test = train_test_split(t_data, test_size=0.2)
    p_train, p_test = train_test_split(p_data, test_size=0.2)
    n_train, n_test = train_test_split(n_data, test_size=0.2)
    
    train_data = pd.concat([t_train, p_train, n_train], axis=0).sample(frac=1, replace=False)
    test_data = pd.concat([t_test, p_test, n_test], axis=0).sample(frac=1, replace=False)
    
    del t_data, p_data, n_data, t_train, t_test, p_train, p_test, n_train, n_test
    
    train_data = train_data.reset_index()
    del train_data['index']
    nouns = train_data['noun']
    label = train_data['label']

    nounvec = loadbyvectors(nouns, ft128)
    label = np.asarray(label)
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf = Pipeline([('norm', StandardScaler()), ('knn',clf)])
    clf.fit(nounvec, label)
        
    test_data = test_data.reset_index()
    del test_data['index']
    
    testnouns= test_data['noun']
    testvec = loadbyvectors(testnouns, ft128)
    testpred = clf.predict(testvec)
    curmean = np.mean(testpred == test_data['label'])
    test_data['pred'] = testpred
    
    f1 = f1_score(test_data['label'], test_data['pred'], average='macro')

    pickle.dump(clf, open('knn_model', 'wb'))

    print("Mean Accuracy : {:.1%}".format(curmean))
    print("Mean F1 score : {:.1%}".format(f1))

    return clf