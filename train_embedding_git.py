# -*- coding: utf-8 -*-
"""
Created on Tue Oct 07 16:25:23 2014

@author: Yang Xuefeng
"""
from __future__ import division
import numpy as np
import cPickle as cp
import sys
import scipy.stats as ss
import bisect
import argparse

class evaluation(object):
    def __init__(self, wl):
        s = set(wl.keys())
        f = np.load(r'D:\SS\ResourceData\antonmys\wordsim353.npz')
        words = f['w']
        score = f['s']
        score = [float(i) for i in score]
        select_index = [i for i in xrange(len(words)) if words[i][0] in s and words[i][1] in s]
        self.words353 = [words[i] for i in select_index]
        self.score353 = [score[i] for i in select_index]
        self.index353 = [(wl.get(i[0],0), wl.get(i[1],0)) for i in self.words353]
        
        f = np.load(r'D:\SS\ResourceData\antonmys\turk771.npz')
        words = f['w']
        score = f['s']
        score = [float(i) for i in score]
        select_index = [i for i in xrange(len(words)) if words[i][0] in s and words[i][1] in s]
        self.words771 = [words[i] for i in select_index]
        self.score771 = [score[i] for i in select_index]
        self.index771 = [(wl.get(i[0],0), wl.get(i[1],0)) for i in self.words771]
        
        f = np.load(r'D:\SS\ResourceData\antonmys\rg65.npz')
        words = f['w']
        score = f['s']
        score = [float(i) for i in score]
        select_index = [i for i in xrange(len(words)) if words[i][0] in s and words[i][1] in s]
        self.words65 = [words[i] for i in select_index]
        self.score65 = [score[i] for i in select_index]
        self.index65 = [(wl.get(i[0],0), wl.get(i[1],0)) for i in self.words65]
        
        f = np.load(r'D:\SS\ResourceData\antonmys\yp130.npz')
        words = f['w']
        score = f['s']
        score = [float(i) for i in score]
        select_index = [i for i in xrange(len(words)) if words[i][0] in s and words[i][1] in s]
        self.words130 = [words[i] for i in select_index]
        self.score130 = [score[i] for i in select_index]
        self.index130 = [(wl.get(i[0],0), wl.get(i[1],0)) for i in self.words130]
        
        f = np.load(r'D:\SS\ResourceData\antonmys\M3k.npz')
        words = f['w']
        score = f['s']
        score = [float(i) for i in score]
        select_index = [i for i in xrange(len(words)) if words[i][0] in s and words[i][1] in s]
        self.words3k = [words[i] for i in select_index]
        self.score3k = [score[i] for i in select_index]
        self.index3k = [(wl.get(i[0],0), wl.get(i[1],0)) for i in self.words3k]
        
        l = cp.load(open(r'D:\SS\ResourceData\antonmys\analogy_g.pkl'))
        l = [i for i in l if i[0] in s and i[1] in s and i[2] in s and i[3] in s]
        self.word_g = l
        self.index_g = [(wl[i[0]],wl[i[1]],wl[i[2]],wl[i[3]]) for i in l]
        index_list = zip(*self.index_g)
        self.index_g_mat = [list(i) for i in index_list]
        l = cp.load(open(r'D:\SS\ResourceData\antonmys\analogy_m.pkl'))
        l = [i for i in l if i[0] in s and i[1] in s and i[2] in s and i[3] in s]
        self.word_m = l
        self.index_m = [(wl[i[0]],wl[i[1]],wl[i[2]],wl[i[3]]) for i in l]
        index_list = zip(*self.index_m)
        self.index_m_mat = [list(i) for i in index_list]
        
        f = np.load(r'D:\SS\ResourceData\antonmys\sent_complete.npz')
        select = []
        
        for i in xrange(len(f['c'])):
            t = [1 for j in f['c'][i] if j in s]
            p = [1 for j in f['s'][i] if j in s]
            if len(t)==5 and 2*len(p)>len(f['s'][i]):
                select.append(i)
        
        #print len(select)
        self.sents = [f['s'][i] for i in select]
        self.candidates = [f['c'][i] for i in select]
        self.answers = [f['a'][i] for i in select]
        self.index_sents = []
        self.index_candidates = []
        self.index_answers = [wl[i] for i in self.answers]
        for i in self.sents:
            t = [wl[j] for j in i if j in s]
            self.index_sents.append(t)
        for i in self.candidates:
            t = [wl[j] for j in i]
            self.index_candidates.append(t)
    def sent_completation(self,epoch,wm):        
        r = []
        for i in xrange(len(self.answers)):
            t = []
            for j in self.index_candidates[i]:
                simi = [self.get_cosine(wm[j,:], wm[k,:]) for k in self.index_sents[i]]
                t.append((j,np.mean(simi)))
            t.sort(key=lambda x:x[1])
            
            r.append(t[-1][0])
        f = [1 if r[i]==self.index_answers[i] else 0 for i in xrange(len(r))] 
        result = sum(f)/len(r)
        return result             
    def word353(self, wm):
        simi = [self.get_cosine(wm[i[0],:], wm[i[1],:]) for i in self.index353]
        r,p = ss.spearmanr(simi, self.score353)
        return r
        
    def turk771(self, wm):
        simi = [self.get_cosine(wm[i[0],:], wm[i[1],:]) for i in self.index771]
        r,p = ss.spearmanr(simi, self.score771)
        
        return r
    def rg65(self,wm):
        simi = [self.get_cosine(wm[i[0],:], wm[i[1],:]) for i in self.index65]
        r,p = ss.spearmanr(simi, self.score65)
        
        return r
    def yp130(self, wm):
        simi = [self.get_cosine(wm[i[0],:], wm[i[1],:]) for i in self.index130]
        r,p = ss.spearmanr(simi, self.score130)
        
        return r
    def m3k(self, wm):
        simi = [self.get_cosine(wm[i[0],:], wm[i[1],:]) for i in self.index3k]
        r,p = ss.spearmanr(simi, self.score3k)
        
        return r
    def analogy(self,wm,t):
        wm_t = wm.transpose()
        if t == 'g':
            a = self.index_g_mat[0]
            b = self.index_g_mat[1]
            c = self.index_g_mat[2]
            d = self.index_g_mat[3]
        elif t == 'm':
            a = self.index_m_mat[0]
            b = self.index_m_mat[1]
            c = self.index_m_mat[2]
            d = self.index_m_mat[3]
        ma = wm[a,:] 
        mb = wm[b,:] 
        mc = wm[c,:]
        m = mb + mc- ma
        l = []
        for i in xrange(len(a)):
            simi = np.dot(m[i,:],wm_t)
            simi[[a[i],b[i],c[i]]] = -1
            l.append(np.argmax(simi))
        r = [1 if d[i]==l[i] else 0 for i in xrange(len(d))]
        r = sum(r)/len(d)
        return r
    def eval_all(self,wm):
        r = []
        r.append(self.word353(wm))
        r.append(self.rg65(wm))
        r.append(self.yp130(wm))
        r.append(self.turk771(wm))
        r.append(self.m3k(wm))
        r.append(self.analogy(wm,'g'))
        r.append(self.analogy(wm,'m'))
        r.append(self.sent_completation(wm))
        return r        
    
        
        
class fine_tuning(object):
    def __init__(self, wl,sr):
        """
        
        """
        self.sr = float(sr)
        self.wl = wl
        
        it = wl.items()
        it = [(i[1],i[0]) for i in it]
        self.lw = dict(it)
    
    def normalization_mat(self, wm):
        a,b = wm.shape
        norms = [np.sqrt(np.sum(np.square(wm[i,:]))) for i in xrange(a)]
        norms = np.array(norms).reshape(a,1)
        wm = wm/norms
        return wm
        
    def normalization_vec(self, v):
        norm = np.sqrt(np.sum(np.square(v)))
        v = v/norm
        return v
    def random_matrix_thres(self, wm):
        np.random.seed(10000)
        a,b = wm.shape
        rm = np.random.uniform(-1,1,(a,b))
        norms = [np.sqrt(np.sum(np.square(rm[i,:]))) for i in xrange(a)]
        norms = np.array(norms).reshape(a,1)
        rm = rm/norms
        rand_index = np.random.randint(0,a-1,300)
        r = []
        rmt = rm.transpose()
        for i in rand_index:
            simi = np.dot(rm[i,:],rmt)
            r.append(np.mean(np.sort(simi)[-3:-1]))
    
        thres = np.mean(r)
        self.thres = thres
           

    def judge(self,i1,i2):
        if i1 < i2:
            return 1
        elif i1 > i2:
            return -1
        else:
            return 0


    def get_local_direction(self, k, wm,simi,kb):
        index = np.argsort(simi)
        simi_sort = simi[index]
        number = wm.shape[0] - bisect.bisect_left(simi_sort, self.thres)
        if number > 300:
            number = 300
        index = index[::-1]
        
        index_dict = {index[i]:i-1 for i in xrange(len(index))}
      
        inter = set.intersection(set(index[0:number]),set(kb))
        kb_bad = set(kb)-inter
        index_bad = set(index[0:number])-inter
        index_bad_sign = [(i,-1) for i in index_bad]
        index_bad_error = [abs(number-index_dict[i]) for i in index_bad]
        kb_bad_sign = [(i,1) for i in kb_bad]
        
        kb_bad_error = [abs(index_dict[kb[i]]-i) for i in xrange(len(kb)) if kb[i] in kb_bad]
        inter_sign_error = [abs(index_dict[kb[i]]-i) for i in xrange(len(kb)) if kb[i] in inter]
        inter_sign = [(kb[i],self.judge(i,index_dict[kb[i]])) for i in xrange(len(kb)) if kb[i] in inter]
        inter_sign = [i for i in inter_sign if i[1]!=0]
        kb_bad_sign.extend(inter_sign)
        kb_bad_sign.extend(index_bad_sign)
        kb_bad_error.extend(index_bad_error)
        kb_bad_error.extend(inter_sign_error)
        error = np.sum(kb_bad_error)
        
        index_sign = zip(*kb_bad_sign)
        
        
        index = list(index_sign[0])
        
        sign = np.array(index_sign[1]).reshape(len(index_sign[1]),1)
        temp = wm[index,:]
        
        result = sign * temp
        return result ,error 
        
        
 
    def get_update(self, k, wm, simi, kb):
        """
        
        """
        result, error = self.get_local_direction(k,wm,simi,kb)
        result = np.mean(result,axis=0).reshape(1,result.shape[1])
        result = self.normalization_vec(result)
        return result, error    

    def get_cosine(self, x, y):
    
        nominator = np.sum( x * y )
        dominator = np.sqrt(np.sum(x*x)) * np.sqrt(np.sum(y*y))
        
        return nominator/dominator
      
    def training(self, wl, wm, kb, eva,evaluate=True):
        epoch = 1
        wm = ft.normalization_mat(wm)
        wm_t = wm.transpose()
        if eva:
            result = eva.eval_all(wm)   
            print result
        error_list = []
        error_list.append(100000000000)
        stop = True
        learning_rate = 0.1
        while(stop):
            count = 0
            l_e = []
            print 'epoch: {}'.format(epoch)
        
            for k in kb.keys():
                count = count + 1
                simi = np.dot(wm[k,:],wm_t)
                update, error = self.get_update(k, wm, simi, kb[k])
                #print error
                l_e.append(error)
                update = update * learning_rate
                wm[k,:] = wm[k,:]+ update 
                wm[k,:] = ft.normalization_vec(wm[k,:])
                sys.stdout.write('{:10d} fin'.format(count))
                sys.stdout.write('\r')
                sys.stdout.flush()
        
            epoch = epoch + 1
            if eva:
                result = eva.eval_all(wm)
                print result
            error = np.mean(l_e)
            error_list.append(error)
            if len(error_list)>2:
                if error_list[-2]-error_list[-1]< error_list[1] * ft.sr:
                    stop = False
        
            print error
        return wm, error_list

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-wm','-word_matrix')
    parser.add_argument('-wl','-word_list')
    parser.add_argument('-kb','-knowledge_base')
    parser.add_argument('-out','-output_file')
    #parser.add_argument('-res','-output_result')
    parser.add_argument('-s','-stop_rate')
    args = parser.parse_args()
    
    name_wl = args.wl
    name_wm = args.wm
    name_kb = args.kb
    name_out = args.out
    #name_res = args.res
    sr = args.s
    
    print 'loading'
    
    kb = cp.load(open(name_kb))
    wl = cp.load(open(name_wl)) 
    wm = np.load(name_wm) 
    norms = np.sqrt(np.sum(np.square(wm),axis=1))
    
    print 'Generating Threshold'
    ft = fine_tuning(wl,sr)
    ft.random_matrix_thres(wm)
    #eva = evaluation(wl)
    print 'Training Start'    
    eva = False
    
    wm,el = ft.training(wl, wm, kb, eva)
    
    np.save(name_out,wm)



        
   
        
    
    


