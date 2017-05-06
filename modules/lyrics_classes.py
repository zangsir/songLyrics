from modules.feature_ext_utils import *
import subprocess
import random,spacy
from modules.ranking import clean_nan_inf
from sklearn.externals import joblib
import gensim
import numpy as np
from gensim.models import Doc2Vec
from gensim.models import Word2Vec

class SongLyrics:
    def __init__(self, clf=joblib.load('model/svm_clf.pkl'), LM=kenlm.LanguageModel('model/train3.lm'), w2v_model=Word2Vec.load('model/rock_train.w2v'), d2v_model=Doc2Vec.load('model/rock_train.d2v'), google_model=gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True), nlp=spacy.load('en'), num_cand=150, online=False):
        self.clf = clf
        self.LM = LM
        self.w2v_model = w2v_model
        self.d2v_model = d2v_model
        self.google_model = google_model
        self.nlp = nlp
        self.online = online
        self.num_cand = num_cand

    def select_best_loglik(self, generated_sents, model):
        d = dict()
        for sent in generated_sents:
            d[sent] = model.score(sent) / np.log2(len(sent) + 1)
        sorted_sents = sorted(d, key=d.__getitem__)
        return sorted_sents[-1]

    def gen_candidates(self):
        command = './ngram -lm model/train3.lm -gen %s' % self.num_cand
        lines = subprocess.getoutput(command)
        lines = lines.split('\n')
        lines = [i for i in lines if i != '']
        return lines

    def gen_first_line(self):
        first_lines = self.gen_candidates()
        return self.select_best_loglik(first_lines, self.LM)

    def sample_pregen_candidates(self, pregen_file):
        f = open(pregen_file, 'r').read().split('\n')
        return random.sample(f, self.num_cand)

    def gen_line(self, prev):
        if self.online:
            candidates = self.gen_candidates()
        else:
            candidates = self.sample_pregen_candidates('txt/better_sents.txt')
        features = ['loglik_norm', 'd2v_dist', 'w2v_dist', 'google_dist', 'rhyme_prev', 'rhyme_current', 'len_prev',
                    'len_cur']
        vectors = np.array([], dtype='float32').reshape(0, len(features))
        for cand in candidates:
            features = np.array(feature_extractor(prev, cand, self.LM, self.w2v_model, self.d2v_model, self.google_model, -1, self.nlp)[:-1])
            vectors = np.vstack([vectors, features])

        vectors = clean_nan_inf(vectors)
        confidence = self.clf.predict_proba(vectors)
        conf_pos = confidence[:, -1]
        best_cand = candidates[np.argmax(conf_pos)]
        return best_cand

    def gen_verses(self, num_verse, nodup=True):
        all_lines = []
        first_line = self.gen_first_line()
        all_lines.append(first_line)
        prev_line = first_line
        for i in range(num_verse):
            new_line = self.gen_line(prev_line)
            if nodup and (new_line in all_lines):
                continue
            all_lines.append(new_line)
            prev_line = new_line
        return all_lines



