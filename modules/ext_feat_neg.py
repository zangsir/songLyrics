from feature_ext_utils import *
import sys
import random
#from vectors.vectors import Vectors

#two strategies: 1. for each sentence, randomly sample another sentence that from another passage; 2. for each sentence, randomly sample a gibberish sentence generated by a ngram model with low loglik.








def build_neg_train(data_file,LM,w2v_model,d2v_model,google_model,label,features,nlp,sample_size):
    # go through the training data set, for each sent, do the negative sampling.
    f=open(data_file,'r').read().split('\n')
    f=[i for i in f if i.strip()!='']
    print (len(f)," sentences in this data")
    all_neg_features= np.array([],dtype='float32').reshape(0,len(features))
    #start=time.time()
    for i in range(sample_size):
        if i%5000==0:
            print(str(i))
        #randomly sample another sentence
        sents=random.sample(f,2)
        neg_features=feature_extractor(sents[0],sents[1],LM,w2v_model,d2v_model,google_model,label,nlp)
        if neg_features==[]:
            continue
        #print (pos_features)
        #print (type(pos_features))
        all_neg_features=np.vstack([all_neg_features,neg_features])
        if i%10000==0:
            #print(all_neg_features)
            #print('-=================')
            outname='neg_feature_'+str(i) + '.npy'
            np.save(outname,all_neg_features)
            all_neg_features= np.array([],dtype='float32').reshape(0,len(features))


    return all_neg_features
 
def main():
    test_run=bool(int(sys.argv[1]))
    d2v_model=Doc2Vec.load('model/rock_train.d2v')
    w2v_model=Word2Vec.load('model/rock_train.w2v')
    LM=kenlm.LanguageModel('model/train3.lm')
    
    
    pos_label=1
    neg_label=0
    features=['loglik_norm','d2v_dist','w2v_dist','google_dist','rhyme_prev','rhyme_current','len_prev','len_cur','label']
    nlp=spacy.load('en')
    if test_run:
        google_model=w2v_model
        data_file='rank/test/lyrics_test_data_clean.txt'
        print('testing...')
    else:
        google_model=gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True) 
        #google_model = Vectors("vectors/GoogleNewsVecs.txt") 
        data_file='rank/train/lyrics_train_data_clean.txt'
        #data_file='txt/gibberish.txt'
    print('loading model finished...')    
    feat_neg=build_neg_train(data_file,LM,w2v_model,d2v_model,google_model,neg_label,features,nlp,50000)
    if not test_run:
        np.savetxt('neg_feature.txt',feat_neg)
        np.save('neg_feature.npy',feat_neg)



if __name__ == '__main__':
    main()



