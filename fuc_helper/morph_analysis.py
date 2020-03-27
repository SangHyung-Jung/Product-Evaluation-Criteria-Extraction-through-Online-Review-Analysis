import os, pickle
import pandas as pd
import time
from konlpy.tag import Twitter

def morph_analy(category):
    tw = Twitter()

    ### crawled csv file, we previously convert to pkl file (middle or small)
    with open('{}_review_file.pkl'.format(category), 'rb') as f:
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
            
    with open('{}_pos_basic.pkl'.format(category),'wb') as f:
        pickle.dump(df_tag_basic,f)

    
