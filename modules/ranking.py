from modules.feature_ext_utils import *
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib
import os


def combine_pickle_data(path,pickle_files,features):
    """given a list of pickled npy files, load them, combine them, return the combined matrix"""
    all_data = np.array([],dtype='float32').reshape(0,len(features))
    
    for p in pickle_files:
        this_data=np.load(path+p)
        all_data=np.vstack([all_data,this_data])
    return all_data


def clean_nan_inf(data):
    inf_indices = np.where(np.isinf(data))
    nan_indices = np.where(np.isnan(data))
    data[nan_indices]=0
    data[inf_indices]=0
    return data


def build_data(size):
    pos_path = 'pos_train_data/'
    neg_path = 'neg_train_data/'
    features=['loglik_norm','d2v_dist','w2v_dist','google_dist','rhyme_prev','rhyme_current','len_prev','len_cur','label']
    pos_npy = [f for f in os.listdir(pos_path) if f.endswith('.npy')]
    neg_npy = [f for f in os.listdir(neg_path) if f.endswith('.npy')]
    #print(pos_npy)
    #print(neg_npy)
    pos_data = combine_pickle_data(pos_path,pos_npy,features)
    #np.save('pos.npy',pos_data)
    idx=np.random.randint(len(pos_data), size=size)
    pos_data = pos_data[idx,:]
    

    neg_data = combine_pickle_data(neg_path,neg_npy,features)
    idx=np.random.randint(len(neg_data), size=size)
    neg_data = neg_data[idx,:]
    
    all_data=np.vstack([pos_data,neg_data])
    all_data = clean_nan_inf(all_data)
    np.savetxt('all_data.txt',all_data)
    np.save('all_data.npy',all_data)
    return all_data


def gen_data_label(all_data):
    data = all_data[:,:-1]
    label = all_data[:,-1].astype(int)
    return data, label



def train_classifier(data,label):
    #X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=0)
    #np.save('test_data.npy',X_test)
    #np.save('test_label.npy',y_test)
    X_train,y_train=data,label
    X_test=np.load('npy/test_data.npy')
    y_test=np.load('npy/test_label.npy')
    features=['loglik_norm','d2v_dist','w2v_dist','google_dist','rhyme_prev','rhyme_current','len_prev','len_cur']
    abalation=0
    if abalation:
        not_use=["d2v_dist",'w2v_dist']
        print("not used:",not_use)
        inds=[features.index(i) for i in features if i not in not_use]
        X_train=X_train[:,inds]
        X_test=X_test[:,inds]

    print('test size:',len(y_test))
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    print ('accuracy:',clf.score(X_test, y_test))
    joblib.dump(clf, 'svm_clf.pkl') 

    # clf = joblib.load('filename.pkl')


def main():
    build=True
    if build:
        all_data=build_data(20000)
    else:
        all_data=np.load('alldata_cnon.npy')
    print('data size:')
    print(all_data.shape)
    data,label=gen_data_label(all_data)
    print('positive:',len([i for i in label if i==1]))
    print('negative:',len([i for i in label if i==0]))

    train_classifier(data,label)

if __name__ == '__main__':
    main()










