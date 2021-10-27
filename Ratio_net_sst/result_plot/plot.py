# coding=UTF-8
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import math
import sys
import numpy as np

def give_lst(path_,type_='test'):
    loss_list=[]
    name='_'.join(path_.split('\\')[-1].split("_")[:-1])
    with open(path_,"r",encoding="utf-8") as r_d:
        for info in r_d.readlines():
            if info.find(type_+'->')!=-1:
                loss=float(info.split("is ")[-1].split(",")[0].strip())
                loss_list.append(loss)
    return loss_list,name

def cal_mean(data):
    data_lst=[]
    for index in range(0,len(data),1):
        if index%10==0:
            data_lst.append(data[index])
    return data_lst
    
def get_method_stru_name(path):
    m_lst=path.split('_')
    if 'ratio' in m_lst:
        name = " ".join(m_lst[:3])
        stru = "[[%s/%s,%s]]"%(m_lst[4],m_lst[5],m_lst[3])
    elif 'RBF' in m_lst:
        name = "RBF"
        stru = m_lst[0]
    else:
        name = "MLP"
        if len(m_lst)==4:
            stru = "[[%s,%s],[%s,%s]]"%(m_lst[0],m_lst[2],m_lst[1],m_lst[2])
        else:
            stru = "[[%s,%s]]"%(m_lst[0],m_lst[1])
    return name,stru
def plot_performance(acc_data,color_list):
    legend=[]
    i=0
    for name,lst in acc_data.items():
        name=name.replace("%","/")
        name=name.replace("MPL","MLP")
        legend.append(name)
        data_lst=[yy for yy in lst]
        #data_lst = cal_mean(y)
        name_i=name.split("_")[0]
        x=[yy for yy in range(len(data_lst))]
        plt.plot(x,data_lst,color_list[i],linestyle=marker_dic[name_i],linewidth=2)
        i+=1
    plt.legend(legend)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel('every step (start from 150th step)')
    plt.ylabel('test accuracy (%)')
    plt.savefig('reslut.png',dpi=600)
    plt.show()


if __name__=="__main__":
    type_=sys.argv[1]
    ckpt_path=glob.glob("Best_all/*log*")
    max_len=0
    acc_data={}
    color_model = list(matplotlib.colors.cnames.values())
    marker_dic={'RBF':':','the ratio net':'-.','MLP':'-'}
    color_list=[]
    for i,ckpt in enumerate(ckpt_path):
        color_list.append(color_model[(i+2)*4])
        lst,name=give_lst(ckpt,type_)
        print("%s： the Best accuracy is %.4f"%(name,max(lst)))
        acc_data[name]=lst[150:1000]
        if len(lst) >max_len:
            max_len=len(lst)
    plot_performance(acc_data,color_list)
