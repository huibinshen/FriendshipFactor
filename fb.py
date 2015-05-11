
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

from sklearn.preprocessing import KernelCenterer
from sklearn import linear_model

from alignf import ALIGNF

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def get_network(data):
    edges = {}
    ids = set()
    all_pairs = np.loadtxt('%s/%s_combined.txt' % (data,data)) 
    n_pair = len(all_pairs)
    for i in range(n_pair): 
        id1 = int(all_pairs[i,0])
        id2 = int(all_pairs[i,1])

        ids.add(id1)
        ids.add(id2)
        if id2 not in edges:
            edges[id2] = set()
        if id1 not in edges:
            edges[id1] = set()
        edges[id2].add(id1)
        edges[id1].add(id2)
    print "Number of nodes in %s data: %d" % (data, len(ids))
    print "Number of edges in %s data: %d" % (data, n_pair)
    return ids, edges

f_ids, f_edges = get_network('facebook')

# Now we have a look of features of nodes in each dataset. The features in twitter is a mass and not categorized at all, as a result, we don't output them here.

#### Features in Facebook

# get the features for the nodes in facebook
fb_res = commands.getoutput('ls facebook/*.edges')
fb_ego_ids = [int(fname[fname.find('/')+1:fname.find('.')]) for fname in fb_res.split('\n')]
fb_featnames = set()

feat_value = {}
for id_ in fb_ego_ids:
    data = open('facebook/' + str(id_) + '.featnames').read()
    for line in data.split('\n'):
        if not line:
            continue
        words = line.split(' ')
        fb_featnames.add(words[1])
        feat_value[int(words[0])] = (words[1], int(words[3]))
        
fb_featnames = sorted(fb_featnames)
ind_count = 0
feat_ind = {}  # featname and its index in the final feature matrix
for featname in fb_featnames:
    feat_ind[featname] = ind_count
    ind_count += 1
#print feat_ind 

n_fb_users = len(f_ids)
n_fb_feat = ind_count

# note all the features are categorical variables
fb_profile = {}  # index is user id, value is profile which is a list of list

for id_ in fb_ego_ids:     
    feat_mat = np.loadtxt('facebook/' + str(id_) + '.feat')
    feat_value = {}  # 1st value in featnames as index, tuple (featname, value) as values 
    data = open('facebook/' + str(id_) + '.featnames').read()
    for line in data.split('\n'):
        if not line:
            continue
        words = line.split(' ')
        feat_value[int(words[0])] = (words[1], int(words[3]))
        
    for i in range(len(feat_mat)):
        uid = int(feat_mat[i,0])-1
        feat_vec = feat_mat[i,1:]
        onfeat_inds = np.where(feat_vec > 0)[0]
        profile = [[] for ii in range(n_fb_feat)]
        for j in onfeat_inds:
            featname, value = feat_value[j]
            ind_inX = feat_ind[featname]
            profile[ind_inX].append(value)
        #print uid, profile
        fb_profile[uid] = profile


# We choose relatively large amount of users and features such that every users
# have those features

def normalize(km):
        n = len(km)
        for i in range(n):
            if km[i,i] == 0:
                km[i,i] = 1e-8
        return km / np.array(np.sqrt(np.mat(np.diag(km)).T * np.mat(np.diag(km))))


feat_ucount = []  # the user count for that feature
for i in range(n_fb_feat):
    count = 0
    for uid, profile in fb_profile.items():
        # uid has that feature value
        if len(profile[i])>0:
            count +=1
    feat_ucount.append(count)

featinds = sorted(range(len(feat_ucount)), key=lambda k: feat_ucount[k])
featinds = featinds[::-1]

uinds = np.array([True]*n_fb_users)
f_inds = []

for ind in featinds:
    tinds = np.array([True]*n_fb_users)
    for i in range(n_fb_users):
        if i not in fb_profile:
            tinds[i] = False
        elif len(fb_profile[i][ind]) == 0:
            tinds[i] = False
    tsum = sum(np.logical_and(uinds,tinds))
    if tsum < 500:
        break
    uinds = np.logical_and(uinds,tinds)
    f_inds.append(ind)


# Get the cleaned data
uu_inds = [i for i in range(n_fb_users) if uinds[i]]
n_users = len(uu_inds)
n_feats = len(f_inds)
print "After cleaning, has %d users, %d features" % (n_users, n_feats)

# build target similarity matrix
print "Construting target similarity matrix"
target = np.zeros((n_users, n_users))
# build target similarity matrix
for i in range(n_users):
    for j in range(i,n_users):
        count = len(f_edges[uu_inds[i]] & f_edges[uu_inds[j]])
        if count > 0:
            target[i,j] = count
            target[j,i] = target[i,j]

target = normalize(target)

np.savetxt('fb_y.txt',target)
# for each feature, construct similarity matrix for every users
fb_sims = []
for i in f_inds:
    print "Construting similarity matrix with feature", fb_featnames[i]
    tsim = np.zeros((n_users, n_users))
    for j in range(n_users):
        for k in range(j,n_users):
            values_j = fb_profile[uu_inds[j]][i]
            values_k = fb_profile[uu_inds[k]][i]
            
            count = len(set(values_j) & set(values_k))
            if j==k:
                if count == 0:
                    print j,k,values_j
            tsim[j,k] = count
            tsim[k,j] = count
    tsim = normalize(tsim)
    fb_sims.append(tsim)

def f_dot(X,Y):
    return sum(sum(X*Y))

#fb_sims = pickle.load(open('fb_sims_test.data','rb'))
w = ALIGNF(fb_sims, target)

w = w / np.linalg.norm(w,1)
f = open('fb_weights.txt','w')
print
for i in range(n_feats):
    print fb_featnames[f_inds[i]], w[i]
    f.write("%s %.4f\n" % (fb_featnames[f_inds[i]], w[i]))
f.close()

###################################################
# Predict friendship strength based on the weights
print 
print "######################################"
print "####### Intimacy regression ##########"

def MSE(pred, real):
    return np.mean((pred-real)**2)

def center(km):
    """ centering km """
    m = len(km)
    I = np.eye(m)
    one = np.ones((m,1))
    t = I - np.dot(one,one.T)/m
    return np.dot(np.dot(t,km),t)

n_folds = 5
tags = np.array([i%n_folds+1 for i in range(n_users)])
reg_err = []
for t in range(1,n_folds+1):
    test = np.array(tags == (t+1 if t+1<6 else 1))
    train = np.array(~test)
    n_train = sum(train)
    n_test = sum(test)

    target_train = center(target[np.ix_(train, train)].copy())
    target_test = center(target[np.ix_(test, test)].copy())

    fb_sims_train = []
    fb_sims_test = []

    for i in range(n_feats):
        # center matrix before alignf
        fb_sims_train.append(center(fb_sims[i][np.ix_(train,train)].copy()))
        fb_sims_test.append(center(fb_sims[i][np.ix_(test,test)].copy()))
        
    ww = ALIGNF(fb_sims_train, target_train)
    # For intimacy regression, we dont't require weights has unit norm
    #ww = ww/np.linalg.norm(ww,1)
    target_pred = np.zeros((n_test,n_test))
    for i in range(n_feats):
        target_pred = target_pred + ww[i]*fb_sims_test[i]

    err_alignf = MSE(target_pred, target_test)

    # compare with ridge regression
    X_train = np.zeros((n_train**2, n_feats))
    X_test = np.zeros((n_test**2, n_feats))
    for i in range(n_feats):
        X_train[:,i] = fb_sims_train[i].flatten()
        X_test[:,i] = fb_sims_test[i].flatten()

    Y_train = target_train.flatten()
    Y_test = target_test.flatten()
    #clf = linear_model.RidgeCV(alphas=[0.01,0.1,1,10,100])
    clf = linear_model.LinearRegression()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    err_ridge = MSE(Y_pred, Y_test)
    print "Fold %d, alignf: %.4f, ridge: %.4f" % (t,err_alignf,err_ridge)
    print "average values in Y_test", np.mean(Y_test)
    print "non-zero in Y_test", len(np.nonzero(Y_test)[0])/len(Y_test)
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
        if uu_inds[i] in f_edges[uu_inds[j]] or uu_inds[j] in f_edges[uu_inds[i]]:
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
    fb_sims_train = []
    fb_sims_test = []
    fb_sims_train_test = []

    for i in range(n_feats):
        # center matrix before alignf
        #kc = KernelCenterer()
        train_km = fb_sims[i][np.ix_(train,train)].copy()
        test_km = fb_sims[i][np.ix_(test,test)].copy()
        test_train_km = fb_sims[i][np.ix_(test,train)].copy()
        #kc.fit(train_km)
        #fb_sims_train.append(kc.transform(train_km))
        #fb_sims_test.append(center(test_km))
        #fb_sims_train_test.append(kc.transform(test_train_km))
        fb_sims_train.append(train_km)
        fb_sims_test.append(test_km)
        fb_sims_train_test.append(test_train_km)

    # add common friends matrix to the input side
    fb_sims_train.append(target[np.ix_(train,train)].copy())
    fb_sims_test.append(target[np.ix_(test,test)].copy())
    fb_sims_train_test.append(target[np.ix_(test,train)].copy())

    n_inputs = len(fb_sims_train)

    ww = ALIGNF(fb_sims_train, target_train)
    # For link prediction, we don't require weights has unit norm
    #ww = ww/np.linalg.norm(ww,2)
    target_pred = np.zeros((n_test,n_test))

    target_pred_train = np.zeros((n_train,n_train))
    target_pred_train_test = np.zeros((n_test, n_train))
    for i in range(n_inputs):
        target_pred = target_pred + ww[i]*fb_sims_test[i]
        target_pred_train = target_pred_train + ww[i]*fb_sims_train[i]
        target_pred_train_test = target_pred_train_test + ww[i]*fb_sims_train_test[i]

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

