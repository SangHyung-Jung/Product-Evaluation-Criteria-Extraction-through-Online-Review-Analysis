import os, pickle, time
import pandas as pd
import numpy as np
from fuc_helper.lda_ import lda_, extract_lda_noun
from fuc_helper.morph_analysis import morph_analy
from gensim.models import FastText

def mk_fasttext_model():
    file_list=[]
    for files in os.listdir(r"where\pos_basic.pkl\located"):
        if files.endswith(".pkl"):
            file_list.append(files)

    with open(file_list[0], 'rb') as f:
        pos = pickle.load(f)
    for file_name in file_list[1:]:
        with open(file_name, 'rb') as f:
            pos1 = pickle.load(f)
        pos += pos1  ### combine all pos_basic file for fasttext model

    ft128 = FastText(pos, size=128, window=5, min_count=10, workers=10, sg=1)
    ft128.save('ft128.model')

if __name__ == "__main__":
    categories = [] #'middle category list'

    for category in categories:
        morph_analy(category)
        tmp_lda = lda_(category)
        extract_lda_noun(tmp_lda, category)
    
    mk_fasttext_model()
    print("U r ready for tagging work")

    ### We need to tag label on lda_noun_no_label.csv files


