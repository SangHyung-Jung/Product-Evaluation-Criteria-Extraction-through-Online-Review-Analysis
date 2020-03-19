# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:35:44 2018

@author: mej09
"""

######################사전 준비 모델######################
#%%형태소 분석
import os, pickle
import pandas as pd
import time
from konlpy.tag import Twitter
tw = Twitter()

os.chdir(r'C:\Users\mej09\Documents\AILab\12주\tagging')
with open('test.pkl', 'rb') as f:
    pos = pickle.load(f)

start_time=time.time()
tag_basic = ['Adjective', 'Verb', 'Adverb', 'Noun']    

df_tag_basic=[]
for i, x in enumerate(pos['review']):
    if not (i+1)%1000:
        now=time.localtime()   
        print(i+1, ' / ', pos['review'].shape[0],'현재시간: %dh  %dm'%(now.tm_hour, now.tm_min), ' time: %d m  %0.2f s'%((time.time()-start_time)//60, (time.time()-start_time)%60))            
    df_re = [y[0] for y in tw.pos(x, norm=True, stem=True) if y[1] in tag_basic]
    if df_re != []:
        df_tag_basic.append(df_re)
        
with open('가전렌탈_pos_basic.pkl','wb') as f:
    pickle.dump(df_tag_basic,f)

#%%pos 합치기 - Fasttext model

#pos 한 파일에 모은 후
file_list=[]
for file in os.listdir(r"C:\Users\mej09\Documents\AILab\12주\pos 명형부동\ldanew"):
    if file.endswith(".pkl"):
        file_list.append(file)

os.chdir(r"C:\Users\mej09\Documents\AILab\12주\pos 명형부동\ldanew")
with open(file_list[0], 'rb') as f:
    pos = pickle.load(f)      
for file_name in file_list[1:]:
    with open(file_name, 'rb') as f:
        pos1 = pickle.load(f)
    pos += pos1

pkl_file_name = 'pos_all.pkl'        
with open(pkl_file_name, 'wb') as f:
    pickle.dump(pos, f)

from gensim.models import FastText  
ft128 = FastText(pos, size=128, window=5, min_count=10, workers=10, sg=1)
ft128.save('ft128.model')

#%%LDA - 중분류
import lda
from sklearn.decomposition import LatentDirichletAllocation
import os
import pickle
import pandas as pd
import numpy as np
import gensim
import time

os.chdir(r'C:\Users\mej09\Documents\AILab\10주')
with open('df_elec_pos.pkl', 'rb') as f:
    pos_data = pickle.load(f)
    
from gensim import corpora
dictionary = corpora.Dictionary(pos_data)
corpus = [dictionary.doc2bow(text) for text in pos_data]

import re
tmp_lda = []
a = 0
start_time=time.time()
while a<10:
    a += 1
    for no_models in range(10):
        no_models += 1
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = no_models, id2word=dictionary, passes=15)
        print('***topic number of model***:',no_models)
        
        for no_topics in range(no_models):
            if no_topics == 0:        
                print('get all topics from model')
            else:
                print('get ',no_topics, 'topics from model')
    
            for no_words in range(20):
                no_words += 1
    
                topics = ldamodel.print_topics(num_topics=no_topics, num_words=no_words)
                for topic in topics:
                    result = re.findall(r'"([^"]*)"', topic[1])
                    tmp_lda.append(result)
                    
    now=time.localtime()                
    print('횟수: ', a , ', 갯수: ', len(tmp_lda))   
    print('현재시간: %dh  %dm'%(now.tm_hour, now.tm_min), ' time: %d m  %0.2f s'%((time.time()-start_time)//60, (time.time()-start_time)%60))            

os.chdir(r'C:\Users\mej09\Documents\AILab\9주')    
pickle.dump(tmp_lda, open('tmp_lda_elec.pkl', 'wb'))    

lda_list = []
for x in tmp_lda:
    for i in x:
        lda_list.append(i)

import collections
counter = collections.Counter(lda_list)
print(counter.most_common())

counts_k = counter.most_common()
topicnouns_k = [counts_k[i][0] for i in range(len(counts_k))]

from konlpy.tag import Twitter, Hannanum, Kkma
tw = Twitter()
hannanum = Hannanum()
kkma = Kkma()

cnouns = []

for i in range(len(topicnouns_k)):
    t = tw.nouns(topicnouns_k[i])
    h = hannanum.nouns(topicnouns_k[i])
    k = kkma.nouns(topicnouns_k[i])
    if h != [] and k != [] and t != []:
        if set(h) == set(h).intersection(set(k),set(t)):
            cnouns += h
            print(h,k,t)
        else: print('not in list',h,k,t)

df = pd.DataFrame(cnouns)
df.columns = ['noun']
df['label'] = np.zeros(len(cnouns))

os.chdir(r'C:\Users\mej09\Documents\AILab\12주\계절가전_가습기')
df.to_csv('계절가전_가습기_lda_new.csv', header=True, encoding='cp949')

##################태깅 실시######################

#%%태깅한 모든 중분류 파일 합치기
file_list=[]
for file in os.listdir(r"C:\Users\mej09\Documents\AILab\12주\pos 명형부동\ldanew"):
    if file.endswith(".csv"): 
        file_list.append(file)

os.chdir(r"C:\Users\mej09\Documents\AILab\12주\pos 명형부동\ldanew")
data_df = pd.read_csv(file_list[0], engine='python')[['noun','label']]        
for file_name in file_list[1:]:
    df = pd.read_csv(file_name, engine='python')[['noun','label']]
    data_df = pd.concat([data_df, df], axis=0, join='outer')

data_df.to_csv('alllda.csv', index=False, encoding = 'cp949')   

#################다르게 태깅된 것 고치기###############

#%%모델 성능 테스트(중복 제거)
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.chdir(r'C:\Users\mej09\Documents\AILab\12주\pos_basic')
ft128 = FastText.load('ft128.model')

means=[]
f1score=[]

os.chdir(r'C:\Users\mej09\Documents\AILab\12주')
data_df = pd.read_csv('alllda2.csv', engine='python')[['noun','label']]
data_df = data_df.drop_duplicates()

i = 0
while i<5:
    i += 1
    t_data = data_df[data_df['label']==2]#16
    p_data = data_df[data_df['label']==1]#161
    n_data = data_df[data_df['label']==0]#783
    
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
    
    def loadbyvectors(data):
        load_x = []
        
        for i in range(len(data)):
            try: tmp = ft128.wv[data[i]]
            except:
                print('not in the vector : ', data[i])
                tmp = np.zeros(128, dtype=np.float32)
            load_x.append(tmp)
        load_x = np.asarray(load_x)      
        return load_x
    
    nounvec = loadbyvectors(nouns)
    label = np.asarray(label)
    clf = KNeighborsClassifier(n_neighbors=3,weights='distance')
    clf = Pipeline([('norm', StandardScaler()), ('knn',clf)])
    clf.fit(nounvec, label)
    
    test_data = test_data.reset_index()
    del test_data['index']
    
    testnouns= test_data['noun']
    testvec = loadbyvectors(testnouns)
    testpred = clf.predict(testvec)
    curmean = np.mean(testpred == test_data['label']) #84%/83.5%
    test_data['pred'] = testpred
    
    from sklearn.metrics import f1_score
    f1 = f1_score(test_data['label'], test_data['pred'],average='macro') #65%, 56%
    
    means.append(curmean)
    f1score.append(f1)

print("Mean Accuracy : {:.1%}".format(np.mean(means)))
print("Mean F1 score : {:.1%}".format(np.mean(f1score)))


#%%모델 학습(중복 허용)
from gensim.models import FastText
import os, pickle
import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.chdir(r'C:\Users\mej09\Documents\AILab\12주\pos_basic')
ft128 = FastText.load('ft128.model')
os.chdir(r'C:\Users\mej09\Documents\AILab\12주')
data_df = pd.read_csv('alllda2.csv', engine='python')[['noun','label']]

nouns = data_df['noun']
label = data_df['label']

def loadbyvectors(data):
    load_x = []
    
    for i in range(len(data)):
        try: tmp = ft128.wv[data[i]]
        except:
            print('not in the vector : ', data[i])
            tmp = np.zeros(128, dtype=np.float32)
        load_x.append(tmp)
    load_x = np.asarray(load_x)      
    return load_x

nounvec = loadbyvectors(nouns)
label = np.asarray(label)

clf = KNeighborsClassifier(n_neighbors=3,weights='distance')
clf = Pipeline([('norm', StandardScaler()), ('knn',clf)])
clf.fit(nounvec, label)

######################평가 기준 추출 모델######################
#%%소분류 형태소 분석
import os, pickle
import pandas as pd
import time
from konlpy.tag import Twitter
tw = Twitter()

os.chdir(r'C:\Users\mej09\Documents\AILab\12주\tagging')
with open('test.pkl', 'rb') as f:
    pos = pickle.load(f)

start_time=time.time()
tag_basic = ['Adjective', 'Verb', 'Adverb', 'Noun']    

df_tag_basic=[]
for i, x in enumerate(pos['review']):
    if not (i+1)%1000:
        now=time.localtime()   
        print(i+1, ' / ', pos['review'].shape[0],'현재시간: %dh  %dm'%(now.tm_hour, now.tm_min), ' time: %d m  %0.2f s'%((time.time()-start_time)//60, (time.time()-start_time)%60))            
    df_re = [y[0] for y in tw.pos(x, norm=True, stem=True) if y[1] in tag_basic]
    if df_re != []:
        df_tag_basic.append(df_re)
        
with open('가전렌탈_pos_basic.pkl','wb') as f:
    pickle.dump(df_tag_basic,f)
    
#%%LDA - 소분류 (count와 함께 저장)
import lda
from sklearn.decomposition import LatentDirichletAllocation
import os
import pickle
import pandas as pd
import numpy as np
import gensim
import time

os.chdir(r'C:\Users\mej09\Documents\AILab\10주')
with open('df_elec_pos.pkl', 'rb') as f:
    pos_data = pickle.load(f)
    
from gensim import corpora
dictionary = corpora.Dictionary(pos_data)
corpus = [dictionary.doc2bow(text) for text in pos_data]

import re
tmp_lda = []
a = 0
start_time=time.time()
while a<10:
    a += 1
    for no_models in range(10):
        no_models += 1
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = no_models, id2word=dictionary, passes=15)
        print('***topic number of model***:',no_models)
        
        for no_topics in range(no_models):
            if no_topics == 0:        
                print('get all topics from model')
            else:
                print('get ',no_topics, 'topics from model')
    
            for no_words in range(10):
                no_words += 1
    
                topics = ldamodel.print_topics(num_topics=no_topics, num_words=no_words)
                for topic in topics:
                    result = re.findall(r'"([^"]*)"', topic[1])
                    tmp_lda.append(result)
                    
    now=time.localtime()                
    print('횟수: ', a , ', 갯수: ', len(tmp_lda))   
    print('현재시간: %dh  %dm'%(now.tm_hour, now.tm_min), ' time: %d m  %0.2f s'%((time.time()-start_time)//60, (time.time()-start_time)%60))            

os.chdir(r'C:\Users\mej09\Documents\AILab\9주')    
pickle.dump(tmp_lda, open('tmp_lda_elec.pkl', 'wb'))    

lda_list = []
for x in tmp_lda:
    for i in x:
        lda_list.append(i)

import collections
counter = collections.Counter(lda_list)
print(counter.most_common())

counts_k = counter.most_common()

import pandas as pd
from konlpy.tag import Twitter, Hannanum, Kkma
import numpy as np
tw = Twitter()
hannanum = Hannanum()
kkma = Kkma()

cnouns = []
count = []

for i in range(len(counts_k)):
    t = tw.nouns(counts_k[i][0])
    h = hannanum.nouns(counts_k[i][0])
    k = kkma.nouns(counts_k[i][0])
    if h != [] and k != [] and t != []:
        if set(h) == set(h).intersection(set(k),set(t)):
            cnouns.append([h[0], counts_k[i][1]])
            print(h,k,t)
        else: print('not in list',h,k,t)

cnouns = pd.DataFrame(cnouns,columns=['noun','count'])

cnouns['label'] = np.zeros(len(cnouns))
os.chdir(r'C:\Users\mej09\Documents\AILab\12주\pos 명형부동')
cnouns.to_csv('컴퓨터주변기기_공유기_lda_new.csv', header=True, encoding='cp949')

#%%태그 예측

model_test_data_file_path = r'C:\Users\mej09\Documents\AILab\12주\pos 명형부동'
os.chdir(model_test_data_file_path)
test = pd.read_csv('저장장치_USB메모리_lda_new.csv', engine = 'python')[['noun','count']]
testnouns= test['noun']
testvec = loadbyvectors(testnouns)
testpred = clf.predict(testvec)
testcount = test['count']
pd.DataFrame({'noun':testnouns, 'pred_label':testpred, 'count':testcount}).to_csv('pred_new.csv',encoding='cp949',index=False)

#%%차원 추출
pred = pd.read_csv('pred_new.csv', engine = 'python')[['noun','pred_label','count']]
labelnoun2 = pred[pred['pred_label']==2][['noun','count']]
labelnoun1 = pred[pred['pred_label']==1][['noun','count']]
labelnoun2['prop'] = labelnoun2['count'] / np.sum(labelnoun2['count'])
labelnoun1['prop'] = labelnoun1['count'] / np.sum(labelnoun1['count'])

final_2 = labelnoun2[labelnoun2['prop']>=0.01]['noun']
final_1 = labelnoun1[labelnoun1['prop']>=0.1]['noun']
