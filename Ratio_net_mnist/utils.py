import numpy as np 
import sys
import random
import math

def load_raw_data(file_,method):
    t_train=[]
    t_test=[]
    labels=[]
    for line in open("minist_auto_encoder","r",encoding="utf-8"):
        path,data=line.strip().split("\t")
        feature=[float(x)for x in data.split("<=>")]
        
        #zero-score
        #if method=='tanh' or method=='sigmoid' or method=='RBF':
        #mean=np.mean(feature)
        #std=np.std(feature,ddof=1)
        #feature=[(x-mean)/std for x in feature]
        
        #Max_Min
        # max_=np.max(feature)
        # min_=np.min(feature)
        # feature=[(x-min_)/(max_-min_) for x in feature]
        
        which,label,im=path.split("/")[-1].split("_")
        label=int(label)
        if label not in labels:
            labels.append(label)
        if which=="train":
            t_train.append([feature,label])
        else:
            t_test.append([feature,label])
    random.shuffle(t_train)
    random.shuffle(t_test)
    print("ok")
    return t_train,t_test,len(t_train[0][0]),len(labels)
    
def to_one_hot(label,label_length):
    ret = [0]*label_length
    ret[label]=1
    return ret

class give_batch():
    def __init__(self,method):
        self.file_='mnist_auto_encoder'
        self.which=method
        
        self.train,self.test,self.input_,self.output_=load_raw_data(self.file_,self.which)
        self.total_train=len(self.train)
        self.ith=0
    
    def do(self, batchsize):
        #restart load batch data
        if (self.ith+1)*batchsize>self.total_train:
            random.shuffle(self.train)
            self.ith=0
            return self.do(batchsize)
        #load batch data
        else:
            ret =self.train[self.ith*batchsize:(self.ith+1)*batchsize]
            feature=[x[0] for x in ret]
            label=[to_one_hot(x[1],self.output_) for x in ret]
            self.ith +=1
            return feature,label
    
    #double times load test batch data
    def do_test_batch(self,batchsize):
        ret = self.test[batchsize:(2)*batchsize]
        feature=[x[0] for x in ret]
        label=[to_one_hot(x[1],self.output_) for x in ret]
        return feature,label
    
    #load all test data
    def do_test_all(self):
        feature=[x[0] for x in self.test]
        label=[to_one_hot(x[1],self.output_) for x in self.test]
        return feature,label
        
if __name__ == '__main__':
    D=give_batch()
    print(D.input_)
    print(D.output_)
    for _ in range(100):
        feature,label=D.do(2000)
        print(D.ith)
    print(feature)
    print(label)
            
            
    
