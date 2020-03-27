import os, pickle, time
import pandas as pd
import numpy as np
from fuc_helper.knn_model import preprocess, knn_model_train, loadbyvectors
from fuc_helper.lda_ import lda_, extract_lda_noun
from fuc_helper.morph_analysis import morph_analy

def preprocess_for_small_category(category):
    morph_analy(category)
    tmp_lda = lda_(category)
    extract_lda_noun(tmp_lda, category)

def predict_tags(category):

    test = pd.read_csv('{}_lda_noun_result_no_label.csv'.format(category), engine = 'python')[['noun','count']]
    testnouns= test['noun']
    testvec = loadbyvectors(testnouns)
    testpred = clf.predict(testvec)
    testcount = test['count']
    pd.DataFrame({'noun':testnouns, 'pred_label':testpred, 'count':testcount}).to_csv('pred_new.csv',encoding='cp949',index=False)

    pred = pd.read_csv('pred_new.csv', engine = 'python')[['noun','pred_label','count']]
    labelnoun2 = pred[pred['pred_label']==2][['noun','count']]
    labelnoun1 = pred[pred['pred_label']==1][['noun','count']]
    labelnoun2['prop'] = labelnoun2['count'] / np.sum(labelnoun2['count'])
    labelnoun1['prop'] = labelnoun1['count'] / np.sum(labelnoun1['count'])

    final_2 = labelnoun2[labelnoun2['prop'] >= 0.01]['noun']
    final_1 = labelnoun1[labelnoun1['prop'] >= 0.1]['noun']
    
    return final_1, final_2

if __name__ == "__main__":

    data_df = preprocess()
    clf = knn_model_train(data_df)
    
    categories = [] # small category list
    for category in categories:
        preprocess_for_small_category(category)
        final_1, final_2 = predict_tags(category)
        final = pd.concat([final_1, final_2])
        print("Evaluation criteria for {} are {}".format(category, final.tolist()))
    
    print("Done")
