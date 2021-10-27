# coding=UTF-8
import os
import glob
import matplotlib.pyplot as plt
import math
import sys
import numpy as np


def give_lst(path_,type_='test'):
    loss_list=[]
    name='_'.join(path_.split('/')[-1].split("_")[:-1])
    with open(path_,"r",encoding="utf-8") as r_d:
        for info in r_d.readlines():
            if info.find(type_+'->')!=-1:
                loss=float(info.split("is ")[-1].split(",")[0].strip())
                loss_list.append(loss)
    return loss_list,name

def cal_mean(data):
    #print(data)
    data_lst=[]
    for index in range(0,2000,1):
        if index <= 2000-50:
            data_lst.append(sum(data[index:index+50])/50)
        else:
            continue
    return data_lst

def plot_performance(acc_data):
    legend=[]
    for name,lst in acc_data.items():
        legend.append(name)
        data_lst=[yy for yy in lst]
        #data_lst = cal_mean(y)
        
        x=[yy for yy in range(len(data_lst))]#[:101]
        #data_lst[:10]=y[:10]
        #x.insert(0,0)
        #data_lst.insert(0,0)
        plt.plot(x,data_lst)
    plt.legend(legend)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel('step')
    plt.ylabel('acc_test (%)')
    plt.savefig('reslut.png')
    plt.show()


if __name__=="__main__":
    type_=sys.argv[1]
    ckpt_path=glob.glob("../logging/Best/*log*")
    max_len=0
    acc_data={}
    for ckpt in ckpt_path:
        lst,name=give_lst(ckpt,type_)
        print("%s： 最高准确率为%.4f"%(name,max(lst)))
        acc_data[name]=lst#[:150]
        if len(lst) >max_len:
            max_len=len(lst)
    plot_performance(acc_data)
