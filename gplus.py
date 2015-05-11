# coding: utf-8

## Measuring the importance of factors for friendship

# *Abstract* 
# empty now.

### Data preprocessing

#### Extracing the network
import sys
import os

import pickle
import commands
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.preprocessing import KernelCenterer
from sklearn import linear_model

from alignf import ALIGNF

def get_network(data):
    mapping = {}  # uid as key, new id as value
    edges = {}
    all_pairs = np.loadtxt('%s/%s_combined.txt' % (data,data)) 
    n_pair = len(all_pairs)
    count = 1
    for i in range(n_pair): 
        id1 = int(all_pairs[i,0])
        id2 = int(all_pairs[i,1])
        if id1 not in mapping:
            mapping[id1] = count
            count += 1
        if id2 not in mapping:
            mapping[id2] = count
            count += 1
        if mapping[id2] not in edges:
            edges[mapping[id2]] = set()
        edges[mapping[id2]].add(mapping[id1])
    print "Number of nodes in %s data: %d" % (data, len(edges))
    print "Number of edges in %s data: %d" % (data, n_pair)
    return edges,mapping


clean = False
if clean:
    #if os.path.exists('g_edges.dict'):
    if False:
        g_edges = pickle.load(open('g_edges.dict','rb'))
        id_map = pickle.load(open('g_idmap.dict','rb'))
    else:
        g_edges, id_map = get_network('gplus')
        pickle.dump(g_edges,open('g_edges.dict','wb'))
        pickle.dump(id_map,open('g_idmap.dict','wb'))


    #### Features in Gplus


    g_res = commands.getoutput('ls gplus/*.edges')
    g_ego_ids = [int(fname[fname.find('/')+1:fname.find('.')]) for fname in g_res.split('\n')]
    g_featnames = set()
    for id_ in g_ego_ids:
        data = open('gplus/' + str(id_) + '.featnames').read()
        for line in data.split('\n'):
            if not line:
                continue
            words = line.split(' ')
            name = words[1]
            name = name[0:name.find(':')]
            g_featnames.add(name)
        break        
    g_featnames = sorted(g_featnames)

    ind_count = 0
    feat_ind = {}  # featname and its index in the final feature matrix
    w = open('gplus_featname.txt','w')
    for featname in g_featnames:
        feat_ind[featname] = ind_count
        ind_count += 1
        w.write("%d %s\n" % (ind_count,featname))
    print feat_ind 

    n_g_feat = ind_count
    if os.path.exists('g_profile.dict'):
        g_profile = pickle.load(open('g_profile.dict','rb'))
        print "g_profile has %d users" % len(g_profile)
    else:
        # note all the features are categorical variables
        g_profile = {}  # index is user id, value is profile which is a list of list
        mapping_list = [{} for i in range(n_g_feat)]
        feat_count = [1]*n_g_feat    
        for id_ in g_ego_ids: 
            feat_mat = np.loadtxt('gplus/' + str(id_) + '.feat', ndmin=2)
            if feat_mat.shape[1] == 1:
                continue
            # 1st value in featnames as index, tuple (featname, value) as values 
            # notice the value here is a string
            feat_value = {}  
            data = open('gplus/' + str(id_) + '.featnames').read()
            for line in data.split('\n'):
                if not line:
                    continue
                words = line.split(' ')
                word = words[1]
                name = word[0:word.find(':')]   
                value = word[(word.find(':')+1):]
                feat_value[int(words[0])] = (name, value)

            for i in range(len(feat_mat)):
                uid = feat_mat[i,0]
                if id_map[uid] not in g_edges:
                    continue
                feat_vec = feat_mat[i,1:]
                onfeat_inds = np.where(feat_vec > 0)[0]
                profile = [[] for ii in range(n_g_feat)]
                for j in onfeat_inds:
                    featname, value = feat_value[j]
                    ind_inX = feat_ind[featname]
                    if value not in mapping_list[ind_inX]:
                        mapping_list[ind_inX][value] = feat_count[ind_inX]
                        feat_count[ind_inX] += 1

                    profile[ind_inX].append(mapping_list[ind_inX][value])
                g_profile[id_map[uid]] = profile

        print "We have all together %d users" % len(g_profile)
        # remove users don't have all the features
        new_g_profile = {}
        for uid, profile in g_profile.items():
            non_empty_count = sum([1 for featlist in profile if featlist])
            if non_empty_count < n_g_feat:
                continue
            elif non_empty_count == n_g_feat:
                new_g_profile[uid] = profile
            else:
                print "Wired thing happend!"

        print "We have %d users have all the features" % len(new_g_profile)
        
        pickle.dump(new_g_profile,open('g_profile.dict','wb'))
        g_profile = new_g_profile

    g_new_edges = {}
    for uid,fans in g_edges.items():
        if uid in g_profile:
            g_new_edges[uid] = fans
    pickle.dump(g_new_edges,open('g_final_edges.dict','wb'))

# now starting the real things
g_profile = pickle.load(open('g_profile.dict','rb'))
g_edges = pickle.load(open('g_final_edges.dict','rb'))
n_users = len(g_profile)
n_feats = 6
g_users = g_edges.keys()

def normalize(km):
        n = len(km)
        for i in range(n):
            if km[i,i] == 0:
                km[i,i] = 1e-8
        return km / np.array(np.sqrt(np.mat(np.diag(km)).T * np.mat(np.diag(km))))

print "Construting target similarity matrix"
# build target similarity matrix
target = np.zeros((n_users,n_users))
for i in range(n_users):
    for j in range(n_users):
        count = len(g_edges[g_users[i]] & g_edges[g_users[j]])
        if count > 0:
            target[i,j] = count
target = normalize(target)
np.savetxt('gplus_y.txt', target)
# for each feature, construct similarity matrix for every users
g_sims = []
for i in range(n_feats):
    print "Construting similarity matrix with feature",i
    tsim = np.zeros((n_users, n_users))
    for j in range(n_users):
        for k in range(j,n_users):
            values_j = g_profile[g_users[j]][i]
            values_k = g_profile[g_users[k]][i]            
            count = len(set(values_j) & set(values_k))
            if j==k:
                if count == 0:
                    print j,k,values_j
            tsim[j,k] = count
            tsim[k,j] = count
    tsim = normalize(tsim)
    g_sims.append(tsim)

f = open('gplus_featname.txt')
data = f.read()
f.close()
featnames = []
for line in data.split('\n'):
    if not line:
        continue
    words = line.split()
    featnames.append(words[1])

# Using alignf to learn the weights

w = ALIGNF(g_sims, target,centering=False)
w = w / np.linalg.norm(w,1)
f = open('g_weights.txt','w')
print
for i in range(n_feats):
    print "%s %.4f" % (featnames[i], w[i])
    f.write("%s %.4f\n" % (featnames[i], w[i]))
f.close()


###################################################
# Predict friendship strength based on the weights
print 
print "######################################"
print "####### Intimacy regression ##########"

def MSE(pred, real):
    return np.mean((pred-real)**2)


n_folds = 5
tags = np.array([i%n_folds+1 for i in range(n_users)])
reg_err = []
for t in range(1,n_folds+1):
    test = np.array(tags == (t+1 if t+1<6 else 1))
    train = np.array(~test)
    n_train = sum(train)
    n_test = sum(test)

    target_train = target[np.ix_(train, train)].copy()
    target_test = target[np.ix_(test, test)].copy()

    g_sims_train = []
    g_sims_test = []

    for i in range(n_feats):
        g_sims_train.append(g_sims[i][np.ix_(train,train)].copy())
        g_sims_test.append(g_sims[i][np.ix_(test,test)].copy())
        
    ww = ALIGNF(g_sims_train, target_train)
    # For intimacy regression, we dont't require weights has unit norm
    #ww = ww/np.linalg.norm(ww,1)
    target_pred = np.zeros((n_test,n_test))
    for i in range(n_feats):
        target_pred = target_pred + ww[i]*g_sims_test[i]

    err_alignf = MSE(target_pred, target_test)

    # compare with ridge regression
    X_train = np.zeros((n_train**2, n_feats))
    X_test = np.zeros((n_test**2, n_feats))
    for i in range(n_feats):
        X_train[:,i] = g_sims_train[i].flatten()
        X_test[:,i] = g_sims_test[i].flatten()

    Y_train = target_train.flatten()
    Y_test = target_test.flatten()
    #clf = linear_model.RidgeCV(alphas=[0.01,0.1,1,10,100])
    clf = linear_model.LinearRegression()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    err_ridge = MSE(Y_pred, Y_test)
    print "Fold %d, alignf: %.4f, ridge: %.4f" % (t,err_alignf,err_ridge)
    print "average values in Ytest", np.mean(Y_test)
    #print ww
    #print clf.coef_
    reg_err.append(err_ridge)
    
print np.mean(reg_err)


print 
print "#####################################"
print "######  Link prediction #############"

def getAccF1(predY, realY):

    pred = predY
    real = realY

    ntp = np.sum(np.logical_and(pred == 1, real==1))
    nfp = np.sum(np.logical_and(pred == 1, real==0))
    ntn = np.sum(np.logical_and(pred == 0, real==0))
    nfn = np.sum(np.logical_and(pred == 0, real== 1))

    if ntp+nfp == 0:
        pre = 0
    else:
        pre = ntp / float(ntp + nfp)
    if ntp+nfn == 0:
        sen = 0
    else:
        sen = ntp / float(ntp + nfn)
    if pre+sen == 0:
        f1 = 0
    else:
        f1 = 2*pre*sen/float(pre+sen)
    acc = (ntp+ntn) / float(ntp+ntn+nfp+nfn)
    return acc,pre,sen, f1


# build target similarity matrix
target_link = np.zeros((n_users, n_users))
for i in range(n_users):
    for j in range(i,n_users):
        #count = len(f_edges[uu_inds[i]] & f_edges[uu_inds[j]])
        #if count > 0:
        if g_users[i] in g_edges[g_users[j]] or g_users[j] in g_edges[g_users[i]]:
            target_link[i,j] = 1
            target_link[j,i] = target_link[i,j]


tete_res = np.zeros((5,4))
tetr_res = np.zeros((5,4))

for t in range(1,n_folds+1):
    #print "Fold",t
    test = np.array(tags == (t+1 if t+1<6 else 1))
    train = np.array(~test)
    #test = np.array(range(n_users))[test]
    #train = np.array(range(n_users))[train]
    n_train = sum(train)
    n_test = sum(test)

    target_train = target_link[np.ix_(train, train)]
    target_test = target_link[np.ix_(test, test)]
    target_train_test = target_link[np.ix_(test, train)]
    g_sims_train = []
    g_sims_test = []
    g_sims_train_test = []

    for i in range(n_feats):
        # center matrix before alignf
        #kc = KernelCenterer()
        train_km = g_sims[i][np.ix_(train,train)].copy()
        test_km = g_sims[i][np.ix_(test,test)].copy()
        test_train_km = g_sims[i][np.ix_(test,train)].copy()
        #kc.fit(train_km)
        #fb_sims_train.append(kc.transform(train_km))
        #fb_sims_test.append(center(test_km))
        #fb_sims_train_test.append(kc.transform(test_train_km))
        g_sims_train.append(train_km)
        g_sims_test.append(test_km)
        g_sims_train_test.append(test_train_km)

    # add common friends matrix to the input side
    g_sims_train.append(target[np.ix_(train,train)].copy())
    g_sims_test.append(target[np.ix_(test,test)].copy())
    g_sims_train_test.append(target[np.ix_(test,train)].copy())

    n_inputs = len(g_sims_train)

    ww = ALIGNF(g_sims_train, target_train)
    # For link prediction, we don't require weights has unit norm
    #ww = ww/np.linalg.norm(ww,2)
    target_pred = np.zeros((n_test,n_test))

    target_pred_train = np.zeros((n_train,n_train))
    target_pred_train_test = np.zeros((n_test, n_train))
    for i in range(n_inputs):
        target_pred = target_pred + ww[i]*g_sims_test[i]
        target_pred_train = target_pred_train + ww[i]*g_sims_train[i]
        target_pred_train_test = target_pred_train_test + ww[i]*g_sims_train_test[i]

    # best threshold is between 0 to 0.1
    # find best threshold on train data
    best_f1 = 0
    best_thr = 0
    for thr in range(10):
        thr = float(thr)/10
        train_pred = target_pred_train.copy()
        train_pred[train_pred > thr] = 1
        train_pred[train_pred <= thr] = 0
        acc, pre, rec, f1 = getAccF1(train_pred, target_train)
        if f1 > best_f1:
            best_thr = thr
            best_f1 = f1

    print "Best thr", best_thr,"with train F1", best_f1
    thr = best_thr
    target_pred[target_pred > thr] = 1
    target_pred[target_pred <= thr] = 0
    target_pred_train_test[target_pred_train_test > thr] = 1
    target_pred_train_test[target_pred_train_test <= thr] = 0

    acc, pre, rec, f1 = getAccF1(target_pred, target_test)
    tete_res[t-1,0] = acc
    tete_res[t-1,1] = f1
    tete_res[t-1,2] = pre
    tete_res[t-1,3] = rec

    acc, pre, rec, f1 = getAccF1(target_pred_train_test, target_train_test)
    tetr_res[t-1,0] = acc
    tetr_res[t-1,1] = f1
    tetr_res[t-1,2] = pre
    tetr_res[t-1,3] = rec

print "Test-Test (Acc, F1, Precision, Recall)"
print np.mean(tete_res,0)
print "Test-Train (Acc, F1, Precision, Recall)"
print np.mean(tetr_res,0)

