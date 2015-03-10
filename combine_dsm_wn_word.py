# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:38:00 2014

@author: Yang Xuefeng
"""
from __future__ import division 
import os
import numpy as np
import cPickle as cp
from collections import defaultdict as dd


def exist_number(w1,w2,s):
    n = [1 for i in s if w1 in i and w2 in i]
    return len(n)
#import gc
#from collections import defaultdict as dd
print 'loading'
s0 = cp.load(open(r'D:\ss\kb\word_set_doc.pkl'))
s1 = cp.load(open(r'D:\ss\kb\word_set_dep.pkl'))
s2 = cp.load(open(r'D:\ss\kb\word_set_senna.pkl'))
s3 = cp.load(open(r'D:\ss\kb\word_set_hlbl.pkl'))
s4 = cp.load(open(r'D:\ss\kb\word_set_skip.pkl'))
s5 = cp.load(open(r'D:\ss\kb\word_set_tomas.pkl'))
s6 = cp.load(open(r'D:\ss\kb\word_set_glove.pkl'))
#s7 = cp.load(open(r'D:\ss\kb\word_set_w2v.pkl'))
s_l = [s0,s1,s2,s3,s4,s5,s6]
#s = set.union(s0,s1,s2,s3,s4,s5,s6)
s = cp.load(open(r'D:\ss\wordlist_glove_50.pkl'))
s = set(s.keys())
words = np.load(r'D:\ss\wordnet_words.npy')
words = [i for i in words if i.isalpha()]
number = cp.load(open(r'D:\ss\kb\numbers.pkl'))
s = set.intersection(s,set(words))
#s = set(words)
data = {i:{} for i in words}
print 'starting'
pathname = r'D:\SS\kb\sense_kb'
names = os.listdir(pathname)
for i in names:
    print i
    name = pathname + '\\' + i
    d = cp.load(open(name))
    #count = 0
    for j in d.keys():
        if j not in s:
            continue
        #count = count + 1
        #print count,j
        #maximum = max([k[1] for k in d[j]])
        leng = len(d[j])
        for k in xrange(len(d[j])):
            if d[j][k] in s:
                if d[j][k] in data[j]:
                    data[j][d[j][k]][0] = data[j][d[j][k]][0]+(leng-k)/leng
                    data[j][d[j][k]][2] = data[j][d[j][k]][2]+1
                else:
                    n = exist_number(j,d[j][k],s_l)
                    data[j][d[j][k]] = [(leng-k)/leng,n,1]
    #print 'deleting'
    #del d
    #print 'garbage clearing'
    #gc.collect()
    print 'done, next'

print 'wordnet'
wn_value = open(r'D:\SS\KB\sense\value_a.txt')
wn_key = open(r'D:\SS\KB\sense\key_wn.txt')
wn_key = [i.strip('\n') for i in wn_key]
for i in xrange(len(wn_key)):
    if wn_key[i] in s:
        line = wn_value.readline()
        if line != '':
            line = line.strip(',\n')
            line = line.split(',')
            line = [k for k in line if k in s]
            for j in line:
                if j in data[wn_key[i]]:
                    data[wn_key[i]][j][0] = data[wn_key[i]][j][0] + 0.75
                    data[wn_key[i]][j][2] = data[wn_key[i]][j][2] + 1
                else:
                    n = exist_number(j,wn_key[i],s_l)
                    data[wn_key[i]][j] = [0.75,n+1,1]
    else:
        line = wn_value.readline()
wn_value = open(r'D:\SS\KB\sense\value_n.txt')
for i in xrange(len(wn_key)):
    if wn_key[i] in s:
        line = wn_value.readline()
        if line != '':
            line = line.strip(',\n')
            line = line.split(',')
            line = [k for k in line if k in s]
            for j in line:
                if j in data[wn_key[i]]:
                    data[wn_key[i]][j][0] = data[wn_key[i]][j][0] + 1
                    data[wn_key[i]][j][2] = data[wn_key[i]][j][2] + 1
                else:
                    n = exist_number(j,wn_key[i],s_l)
                    data[wn_key[i]][j] = [1,1+n,1]
    else:
        line = wn_value.readline()
wn_value = open(r'D:\SS\KB\sense\value_v.txt')
for i in xrange(len(wn_key)):
    if wn_key[i] in s:
        line = wn_value.readline()
        if line != '':
            line = line.strip(',\n')
            line = line.split(',')
            line = [k for k in line if k in s]
            for j in line:
                if j in data[wn_key[i]]:
                    data[wn_key[i]][j][0] = data[wn_key[i]][j][0] + 0.5
                    data[wn_key[i]][j][2] = data[wn_key[i]][j][2] + 1
                else:
                    n = exist_number(j,wn_key[i],s_l)
                    data[wn_key[i]][j] = [0.5,n+1,1]
    else:
        line = wn_value.readline()
wn_value = open(r'D:\SS\KB\sense\value_r.txt')
for i in xrange(len(wn_key)):
    if wn_key[i] in s:
        line = wn_value.readline()
        if line != '':
            line = line.strip(',\n')
            line = line.split(',')
            line = [k for k in line if k in s]
            for j in line:
                if j in data[wn_key[i]]:
                    data[wn_key[i]][j][0] = data[wn_key[i]][j][0] + 0.75
                    data[wn_key[i]][j][2] = data[wn_key[i]][j][2] + 1
                else:
                    n = exist_number(j,wn_key[i],s_l)
                    data[wn_key[i]][j] = [0.75,1+n,1]
    else:
        line = wn_value.readline()
wn_value = open(r'D:\SS\KB\sense\value_s.txt')
for i in xrange(len(wn_key)):
    if wn_key[i] in s:
        line = wn_value.readline()
        if line != '':
            line = line.strip(',\n')
            line = line.split(',')
            line = [k for k in line if k in s]
            for j in line:
                if j in data[wn_key[i]]:
                    data[wn_key[i]][j][0] = data[wn_key[i]][j][0] + 0.75
                    data[wn_key[i]][j][2] = data[wn_key[i]][j][2] + 1
                else:
                    n = exist_number(j,wn_key[i],s_l)
                    data[wn_key[i]][j] = [0.75,1+n,1]
    else:
        line = wn_value.readline()
print 'calculate nummber'
#d = {i:{} for i in words}
for i in data.keys():
    for j in data[i].keys():
        if data[i][j][2]>1:
        
            data[i][j] = data[i][j][0] / data[i][j][1]
        else:
            data[i][j] = 0
        
			

print 'processing numbers'
for i in data.keys():
    if i not in number:
        data[i] = {k:data[i][k] for k in data[i].keys() if k not in number and data[i][k]>=0.1}
print 'output'
fk = open(r'D:\ss\kb\sense\key_word.txt','w')
fv = open(r'D:\ss\kb\sense\value_word.txt','w')
for i in data.keys():
    
    fk.write(i)
    fk.write('\n')
    items = data[i].items()
    #items = [k for k in items if k[1]>1.01]
    items.sort(key = lambda x:x[1])
    items = items[::-1]
    #if len(items) > 200:
    #    items = items[0:200]
    for p in items:
        fv.write(p[0]+':'+str(p[1])+',')
    fv.write('\n')
    print i, len(items)
fk.close()
fv.close()
#f_d = {i:data[i] for i in data.keys() if len(data[i])!=0}
#f_d = {i:f_d[i].items() for i in f_d.keys()}
#for i in f_d.keys():
#    f_d[i].sort(key=lambda x:x[1])
#del data
#gc.collect()

#cp.dump(f_d,open(r'D:\ss\kb\sense\dsm.pkl','w'))
        
        