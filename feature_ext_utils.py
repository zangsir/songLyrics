import kenlm,re
import gensim,os
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
import pronouncing

import spacy



def is_rhyme(word1,word2):
    word1=re.sub(u'[^A-Za-z]','',word1)
    word2=re.sub(u'[^A-Za-z]','',word2)
    return int((word1 in pronouncing.rhymes(word2)) or word1==word2)
    
def clean_word(word): 
    return re.sub(u'[^A-Za-z]','',word)

def is_rhyme_current(sent):
    sent=sent.split(' ')
    last_word=sent[-1]
    for w in sent[:-1]:
        if is_rhyme(w,last_word):
            return 1
    return 0
    
    
def get_loglik_norm(sent,model):
    
    #loglik_norm=model.score(sent)/len(sent.split(' '))
    loglik_norm=model.score(sent)/np.log2(len(sent.split(' '))+1)
    return loglik_norm

def get_d2v_dist(sent1,sent2,model):
    
    v1=model.infer_vector(sent1.split(' '))
    v2=model.infer_vector(sent2.split(' '))
    v1 = np.nan_to_num(v1)
    v2 = np.nan_to_num(v2)
    return cosine(v1,v2)



def make_feature_vec(words, model, num_features,nlp):
    # average all of the word vectors in a sentence
    #words=nlp(words)
    #words=[token.lemma_ for token in words]
    words=words.split(' ')
    #print (words)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0
    for word in words:
        word = clean_word(word)
        if word not in model:
            #print(word)
            
            word = nlp(word)
            try:
                word = [i.lemma_ for i in word][0]
            except IndexError:
                continue

        if word not in model:
            continue

        featureVec = np.add(featureVec,model[word])
        nwords = nwords + 1
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def get_w2v_dist(sent1,sent2,model,nlp):
    
    v1=make_feature_vec(sent1,model,100,nlp)
    v2=make_feature_vec(sent2,model,100,nlp)
    v1 = np.nan_to_num(v1)
    v2 = np.nan_to_num(v2)
    return cosine(v1,v2)

def get_google_w2v_dist(sent1,sent2,model,nlp):
    v1=make_feature_vec(sent1,model,300,nlp)
    v2=make_feature_vec(sent2,model,300,nlp)
    v1 = np.nan_to_num(v1)
    v2 = np.nan_to_num(v2)
    return cosine(v1,v2)



def feature_extractor(current,prev,LM,w2v_model,d2v_model,google_model,label,nlp):
    loglik_norm=get_loglik_norm(current,LM)#LM='train3.lm'
    d2v_dist=get_d2v_dist(prev,current,d2v_model)
    w2v_dist=get_w2v_dist(prev,current,w2v_model,nlp)
    #d2v_dist=w2v_dist
    google_w2v_dist=get_google_w2v_dist(prev,current,google_model,nlp)
    #google_w2v_dist=w2v_dist
    rhyme_prev=is_rhyme(prev.split(' ')[-1],current.split(' ')[-1])
    rhyme_current=is_rhyme_current(current)
    num_words_cur=np.log(len(current))
    num_words_prev=np.log(len(prev))
    label=label
    return np.array([loglik_norm,d2v_dist,w2v_dist,google_w2v_dist,rhyme_prev,rhyme_current,num_words_prev,num_words_cur,label],dtype='float32')

def extract_features_pos(passage,LM,w2v_model,d2v_model,google_model,label,nlp):
    """extract feature from one passage"""
    # a passage is a consecutive set of lines without a blank line in between. we extract features with these pairs 
    # of lines as prev and next lines. they're a more coherent unit. The passages is obtained by methods above, 
    # namely, splitting the training file by '\n\n'
    line_list=passage.split('\n')
    line_list=[i for i in line_list if i!='']
    #print (line_list)
    if len(line_list)<=1:
        #print('len is 1')
        return []
    features=['loglik_norm','d2v_dist','w2v_dist','rhyme_prev','rhyme_current','len_prev','len_cur','label']
    pos_feature_vec=[]
    for i in range(1,len(line_list)):
        #extract features from the current and prev line
        prev=line_list[i-1]
        current=line_list[i]
        #print('prev,current:')
        #print(prev,current)
        #features
        features=feature_extractor(current,prev,LM,w2v_model,d2v_model,google_model,label,nlp)
        pos_feature_vec.append(features)
    return np.array(pos_feature_vec)
        
        
        
