import os, re, pickle, gensim, time
import pandas as pd
import numpy as np
from gensim import corpora
import collections
from konlpy.tag import Twitter, Hannanum, Kkma

### extract topic from pos_basic per category
def lda_(category):
    with open('{}_pos_basic.pkl'.format(category), 'rb') as f:
        pos_data = pickle.load(f)
    
    dictionary = corpora.Dictionary(pos_data)
    corpus = [dictionary.doc2bow(text) for text in pos_data]

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

    pickle.dump(tmp_lda, open('tmp_lda_{}.pkl'.format(category), 'wb'))

    return tmp_lda

### extract noun from lda result
def extract_lda_noun(tmp_lda, category):
    lda_list = []
    for x in tmp_lda:
        for i in x:
            lda_list.append(i)

    counter = collections.Counter(lda_list)
    print(counter.most_common())

    counts_k = counter.most_common()
    topicnouns_k = [counts_k[i][0] for i in range(len(counts_k))]
    
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

    df.to_csv('{}_lda_noun_result_no_label.csv'.format(category), header=True, encoding='cp949')
