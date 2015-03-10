# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:04:19 2014

@author: Yang Xuefeng
"""
from __future__ import division
#import os
import sys
import bisect
#import matplotlib.pyplot as plt
import numpy as np
import cPickle as cp
from nltk.corpus import names
from nltk.corpus import wordnet as wn
import argparse
s = cp.load(open(r'D:\ss\wordlist_glove_50.pkl'))
s = set(s.keys())
parser = argparse.ArgumentParser()
parser.add_argument('-wm','-word_matrix',help='the word matrix file directory')
parser.add_argument('-wl','-word_list', help='the word list file directory')
parser.add_argument('-out','-output_file')
args = parser.parse_args()
print args.wm
print args.wl
print args.out

print 'loading'
# names to remove
ne = names.words()
ne_low = [i.lower() for i in ne]
ne.extend(ne_low)
ne_s = set(ne)
# wordnet lemma
lm = wn.all_lemma_names()
lm = [i for i in lm]
lm_s = set(lm)
lm_s = set.intersection(lm_s,s)


matrix_name = args.wm
list_name = args.wl
wm = np.load(matrix_name)
wl = cp.load(open(list_name))

# remove ne and choose the word in wordnet
nwl = [i for i in wl.keys() if i in lm_s and i not in ne_s]
print 'word number :{}'.format(len(nwl))
wm_index = [wl[i] for i in nwl]
dict_nwl = [(nwl[i],i) for i in xrange(len(nwl))]
wl = dict(dict_nwl)
dict_nwl_inverse = [(i[1],i[0]) for i in dict_nwl]
lw = dict(dict_nwl_inverse)
wm = wm[wm_index,:]
a,b = wm.shape
print 'shape: {} {}'.format(a,b)


print 'normalization'
# normalization
norms = [np.sqrt(np.sum(np.square(wm[i,:]))) for i in xrange(a)]
norms = np.array(norms).reshape(a,1)
wm = wm/norms
wmt = wm.transpose()
print 'random matrix testing'
# get the threshold
rm = np.random.uniform(-1,1,(a,b))
norms = [np.sqrt(np.sum(np.square(rm[i,:]))) for i in xrange(a)]
norms = np.array(norms).reshape(a,1)
rm = rm/norms
rand_index = np.random.randint(0,a-1,1000)
r = []
rmt = rm.transpose()
for i in rand_index:
    simi = np.dot(rm[i,:],rmt)
    r.append(np.mean(np.sort(simi)[-6:-1]))
    
thres = np.mean(r)
print 'thres {}'.format(thres)
#v = [-1,1,0,4000]

d = {}
count = 0
for i in wl.items():
    simi = np.dot(wm[i[1],:],wmt)
    #print i
    #plt.axis(v)
    #plt.hist(simi,bins=100,range=(-1,1))
    #plt.show()
    index = np.argsort(simi)
	#index = index[::-1]
    simi_sort = simi[index]
    number = bisect.bisect_right(simi_sort,thres)
    #print i, a-number
    maxnum = simi_sort[-2]
    if a-number > 250:
		d[i[0]] = [lw[j] for j in index[-251:-1]]
		d[i[0]] = d[i[0]][::-1]
    #if 500 >= a-number > 50:
        
    #    d[i[0]] = [(lw[j],simi[j]/maxnum) for j in index[number-a:-1]]
    else:
        #l = [(lw[j],1) for j in index[number-a:-1]]  
        l = [lw[j] for j in index[number-a:-1]]
        l = l[::-1]
        #l.extend(l_ext)
        d[i[0]] = l
        #print 12131312, len(l)
    sys.stdout.write('{:3.2f} finished'.format(count/a))
    sys.stdout.write('\r')
    sys.stdout.flush()
    #os.system('cls' if os.name == 'nt' else 'clear')
    count =  count + 1
out_name = args.out
cp.dump(d,open(out_name,'w'))