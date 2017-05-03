from feature_ext_utils import *
#import time

def unit_test(test_pas,LM,w2v_model,d2v_model,google_model,pos_label):
    features = extract_features(test_pas,LM,w2v_model,d2v_model,google_model,pos_label)
    for i in features:
        print(i)

def test(LM,w2v_model,d2v_model,google_model,pos_label):
    test_pas_real="""I wanna love you like it ain't no secret
Like I'm not ashamed to show
Nor would I ever, oh never never
Oh never let you go
I'll never let you go"""

    test_pas_gib="""Like I'm a broken relates
No he saves the jams
'Cause I'm rolling stone? love, don't you are near
This can make this
Wearing last place that I get the chance to put on the cell phone
Lord, attack is clear"""

    unit_test(test_pas_real,LM,w2v_model,d2v_model,google_model,pos_label)
    unit_test(test_pas_gib,LM,w2v_model,d2v_model,google_model,pos_label)


def extract_positive_features(data_file,LM,w2v_model,d2v_model,google_model,pos_label,features,nlp):
    #f=open('rank/test/lyrics_test_data_clean.txt','r').read().split('\n\n')
    f=open(data_file,'r').read().split('\n\n')
    f=[i for i in f if i.strip()!='']
    print (len(f)," paragraphs in this data")
    all_pos_features= np.array([],dtype='float32').reshape(0,len(features))
    #start=time.time()
    for i in range(len(f)):
        passage=f[i]
        if i%2000==0:
            print(str(i))

        #print (passage)
        pos_features=extract_features(passage,LM,w2v_model,d2v_model,google_model,pos_label,nlp)
        if pos_features==[]:
            continue
        #print (pos_features)
        #print (type(pos_features))
        all_pos_features=np.vstack([all_pos_features,pos_features])
        if i%10000==0:
            outname='pos_feature_'+str(i) + '.npy'
            #np.savetxt('pos_feature',feat_pos)
            np.save(outname,all_pos_features)
            all_pos_features= np.array([],dtype='float32').reshape(0,len(features))


    return all_pos_features




def main():

    d2v_model=Doc2Vec.load('rock_train.d2v')
    #d2v_model=Word2Vec.load('rock_train.w2v')
    w2v_model=Word2Vec.load('rock_train.w2v')
    LM=kenlm.LanguageModel('train3.lm')
    google_model=gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)  
    #google_model=w2v_model
    print('loading model finished...')
    pos_label=1
    neg_label=0
    features=['loglik_norm','d2v_dist','w2v_dist','google_dist','rhyme_prev','rhyme_current','len_prev','len_cur','label']
    nlp=spacy.load('en')
    #test(LM,w2v_model,d2v_model,google_model,pos_label)
    #print('-----------------')
    #test(w2v_model,d2v_model,google_model,neg_label)
    #data_file='rank/test/lyrics_test_data_clean.txt'
    data_file='rank/train/lyrics_train_data_clean.txt'
    feat_pos=extract_positive_features(data_file,LM,w2v_model,d2v_model,google_model,pos_label,features,nlp)
    np.savetxt('pos_feature.txt',feat_pos)
    np.save('pos_feature.npy',feat_pos)



if __name__ == '__main__':
    main()
        
        
