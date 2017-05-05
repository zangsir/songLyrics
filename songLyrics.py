from feature_ext_utils import *
import subprocess,sys
from sklearn.externals import joblib
import random,spacy
from ranking import clean_nan_inf
import time

# first we use ngram LM to generate K lines. 
# then we use ranking to produce the best next line given the first line. 


def select_best_loglik(generated_sents,model):
    d=dict()
    for sent in generated_sents:
        d[sent]=model.score(sent)/np.log2(len(sent)+1)
    sorted_sents=sorted(d, key=d.__getitem__)
    return sorted_sents[-1]

def gen_candidates(num_cand):
    command = './ngram -lm model/train3.lm -gen %s'%num_cand
    #print(command)
    lines = subprocess.getoutput(command)
    #print(lines)
    lines = lines.split('\n')
    lines = [i for i in lines if i!='']

    return lines

def gen_first_line(num_cand,LM):
    first_lines=gen_candidates(num_cand)
    return select_best_loglik(first_lines,LM)

def sample_pregen_candidates(num_cand,pregen_file):
    f=open(pregen_file,'r').read().split('\n')
    return random.sample(f,num_cand)



def gen_line(prev,num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp,online=False):
    #print('gen one line...')
    if online:
        #print('generating online..')
        candidates=gen_candidates(num_cand)
    else:
        candidates=sample_pregen_candidates(num_cand,'txt/better_sents.txt')#faster than generating online using SRILM, also filtered out lower half of loglik sents
    #given prev line, classify all candidates as appropriate next line or not, then use probability to select best
    #print('feature extraction...')
    features=['loglik_norm','d2v_dist','w2v_dist','google_dist','rhyme_prev','rhyme_current','len_prev','len_cur']
    vectors= np.array([],dtype='float32').reshape(0,len(features))
    for cand in candidates:
        features=np.array(feature_extractor(prev,cand,LM,w2v_model,d2v_model,google_model,-1,nlp)[:-1])
        vectors=np.vstack([vectors,features])
    #print(vectors)    
    #vectors=np.array(vectors)
    vectors=clean_nan_inf(vectors)
    #print('ranking...')
    confidence=clf.predict_proba(vectors)
    conf_pos=confidence[:,-1]
    #print(conf_pos)
    best_cand=candidates[np.argmax(conf_pos)]
    #print(np.max(conf_pos))
    return best_cand







def gen_verses(num_verse,num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp,nodup=True,online=False):
    all_lines=[]
    first_line=gen_first_line(num_cand,LM)
    all_lines.append(first_line)
    #print(first_line)
    prev_line=first_line
    #print('no duplicates:',nodup)
    for i in range(num_verse):
        new_line=gen_line(prev_line,num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp,online)
        if nodup and (new_line in all_lines):
            continue
        all_lines.append(new_line)
        prev_line=new_line
    return all_lines


def stdout_song_lyrics(num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp):
    #for testing, gen lyrics to stdout, not used in final production
    verses=gen_verses(num_verse,num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp)
    choruses=gen_verses(num_chorus,num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp)
    print('==================================================')
    print('Verses:\n')
    for l in verses:
        print(l)
    print("\n\nChorus:\n")
    for l in choruses:
        print(l)



def main():
    start=time.time()
    num_songs=sys.argv[1]
    online=int(sys.argv[2])
    d2v_model=Doc2Vec.load('model/rock_train.d2v')
    w2v_model=Word2Vec.load('model/rock_train.w2v')
    LM=kenlm.LanguageModel('model/train3.lm')
    nlp=spacy.load('en')
    google_model=gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
    #google_model = Vectors("vectors/GoogleNewsVecs.txt")
    clf=joblib.load('model/svm_clf.pkl')
    num_verse=random.randint(5,11)
    num_chorus=4
    num_cand=150
    print('finished loading models...')
    f=open('lyrics_gen.txt','a')
    print('loading costs ',time.time()-start)
    start=time.time()
    for i in range(int(num_songs)):
        print(i)
        verses=gen_verses(num_verse,num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp,online=online)
        choruses=gen_verses(num_chorus,num_cand,clf,LM,w2v_model,d2v_model,google_model,nlp,False,online=online)
        verses='\n'.join(verses)
        choruses='\n'.join(choruses)
        song='\n===========\nVerses:\n%s \n\nChorus:\n%s'%(verses,choruses)
        f.write(song+'\n\n++++++++++++++\n')
    f.close()
    print (time.time()-start)


if __name__ == '__main__':
    main()

