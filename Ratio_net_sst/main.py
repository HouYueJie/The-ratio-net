import tensorflow.compat.v1 as tf 

import numpy as np 
import neural_networks as N
from utils import give_batch
import config as C
import time    
import logging
import os
import glob

tf.disable_eager_execution() 
os.environ["CUDA_VISIBLE_DEVICES"]='-1' #if use gpu, please replace -1 to GPU model number.

def get_struc_name(structure,method='None'):
    if not isinstance(structure,list):
        name = str(structure)
    else:
        if len(structure)==1 and len(structure[0])==2:
            name = str(structure[0][0])+'_'+structure[0][1]
            return name
        elif method=='MLP':
            name_lst=[]
            for i in range(len(structure)):
                name_lst.append(str(structure[i][0]))
            name_lst.append(structure[0][1])
            name = '_'.join(name_lst)
            return name
        elif method=='Pade':
            name_lst=[]
            for i in range(len(structure)):
                name_lst.append(str(structure[i][0]))
            name = '_'.join(name_lst)
            return name

def create_ckpt_document(save_path,stru_name,net_name):
    if isinstance(stru_name,list):
        stru_name = str(stru_name[0])
    ckpt_save_path = save_path+'/'+stru_name+'_'+net_name
    if not os.path.exists(ckpt_save_path):
        print("First training model , then to build ckpt path：%s"%(ckpt_save_path))
        os.mkdir(ckpt_save_path)
    else:
        new_train=input("Did you delete ckpt weights? (yes or no)")
        file_lst=glob.glob(ckpt_save_path+'/*')
        if new_train=="yes" and file_lst!=[]:
            for file in file_lst:
                os.remove(file)
    return ckpt_save_path

def show_parameter_count(loging,variables):
    print("incount")
    total_parameters = 0
    for variable in variables:
        name = variable.name
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim
        info_varibale = '{}: {} ({} parameters)'.format(name,shape,variable_parametes)
        loging.info(info_varibale)
        total_parameters += variable_parametes
    info_var_num='Total: {} parameters'.format(total_parameters)
    loging.info(info_var_num)
    print(info_var_num)
    loging.info('-----------------------------------')

class the_net():
    def __init__(self,train_config,stru_config,name,loggin):
        self.net_name=stru_config["which"]
        self.loggin=loggin
        self.loggin.info("now initialize the net with para:")
        for item,value in train_config.items():
            self.loggin.info(str(item))
            self.loggin.info(str(value))
            self.loggin.info("-----------------------------")
        
        self.save_path=train_config["CKPT"]+"_"+self.net_name
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size = train_config["BATCHSIZE"]
        self.max_iter = train_config["MAX_ITER"]
        self.epoch_save=train_config["EPOCH_SAVE"]
        self.step_each_iter=train_config['STEP_EACH_ITER']
        self.step_show_train=train_config['STEP_SHOW_train']
        self.step_show_test=train_config['STEP_SHOW_test']
        
        self.global_steps = tf.Variable(0, trainable=False)  
        self.stru_config=stru_config

        self.model_struc_name=name
        self.D=give_batch("sst2")
        
        self.sess=tf.Session()
        print("openning sess")
        self.loggin.info("openning Sess")
        self.loggin.info('-----------------------------------')
        
        self.build_net()
        self.build_loss()
        print("building net")
        self.loggin.info("building net")
                
        self.build_opt()
        print("building opt")
        self.loggin.info("building opt")
        
        self.saver=tf.train.Saver(max_to_keep=3)
        self.ckpt_save_path = create_ckpt_document(self.save_path,self.model_struc_name,self.net_name)
        print("build ckpt save path")
        
        self.initialize()
        print("net initializing")
        self.loggin.info("net initializing")


    def build_net(self):
        self.target=tf.placeholder(tf.float32, [None, self.D.output])
        if self.net_name=="MLP":
            self.y = N.MLP(self.stru_config,self.D.input,self.D.output)
        elif self.net_name=="Pade":
            self.y = N.Pade(self.stru_config,self.D.input,self.D.output)
        elif self.net_name=="RBF":
            self.y = N.RBF(self.stru_config,self.D.input,self.D.output)
        else:
            print("Understand model type, please choice('MLP','Pade','RBF')！")
            exit()
        variables=self.y.get_trainable_variables()
        print("The %s model's parameters:\n %s\n"%(self.net_name,variables))
        show_parameter_count(self.loggin,variables)

###########################################
    def build_loss(self):
        cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y.value,
                                                              labels=self.target,
                                                              name='xentropy_per_example')
        self.loss = tf.reduce_mean(cross_entropy, name='loss')

 
    def cal_acc(self,pre,real):
        right=0
        
        for p,r in zip(pre,real):
            p=list(p)
            r=list(r) 
            if p.index(max(p))==r.index(max(r)):
                right +=1
        return float(right)/len(real)
        
    def build_opt(self):
        self.learning_rate_d =self.learning_rate
        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate_d).minimize(self.loss,global_step=self.global_steps)

    def initialize(self):
        ckpt=tf.train.latest_checkpoint(self.save_path)
        if ckpt!=None:
            self.saver.restore(self.sess,ckpt)
            print("init from ckpt ")
            self.loggin.info("init from ckpt ")
        else:
            self.sess.run(tf.global_variables_initializer())
        
    def train(self):
        st=time.time()
        for epoch in range(self.max_iter):
            print("+++++++++++++++++++++++++++++++++++++++++++++train epoch %s of total %s epoches+++++++++++++++++++++++++++++++++++++++++++++"%(epoch,self.max_iter))
            self.loggin.info("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            for step in range(self.step_each_iter):
                feature,label=self.D.do(self.batch_size)
                loss,_,gs,logit=self.sess.run([self.loss,self.opt,self.global_steps,self.y.value],\
                                        feed_dict={self.y.input:feature,self.target:label})
                if (step+1)%self.step_show_train==0:
                    acc=self.cal_acc(logit,label)
                    if (step+1)%(self.step_show_train*5)==0:
                        print("loss %.4f, in epoch %s, in step %s, in global step %s,\
                        acc is %s, taks %s seconds"%(loss,epoch,step,gs,acc,time.time()-st))
                    self.loggin.info("loss %s, in epoch %s, in step %s, in global step %s, acc is %s,\
                    taks %s seconds"%(loss,epoch,step,acc,acc,time.time()-st))
                    st=time.time()
                if (step+1)%self.step_show_test==0:
                    feature,label=self.D.do_test_batch(self.batch_size)
                    loss,logit=self.sess.run([self.loss,self.y.value],\
                                        feed_dict={self.y.input:feature,self.target:label})
                    acc=self.cal_acc(logit,label)
                    if (step+1)%(self.step_show_test*5)==0:
                        print("test->loss %.4f, in epoch %s, in step %s, in global step %s,\
                        acc is %s, taks %s seconds"%(loss,epoch,step,gs,acc,time.time()-st))
                    self.loggin.info("test->loss %s, in epoch %s, in step %s, in global step %s, \
                    acc is %s, taks %s seconds"%(loss,epoch,step,gs,acc,time.time()-st))
                    st=time.time()
            if (epoch+1)%self.epoch_save==0:
                self.saver.save(self.sess, self.ckpt_save_path+"/check.ckpt")
                print("Model saved in path: %s in epoch %s. acc is %s, loss is %s" % (self.ckpt_save_path,epoch,acc,loss))
                self.loggin.info("Model saved in path: %s in epoch %s. acc is %s, loss is %s" % (self.ckpt_save_path,epoch,acc,loss))
            

if __name__ == '__main__':
    which=sys.argv[1]
    if which=="MLP":
        logger = logging.getLogger(C.MLP_config["which"])
        logger.setLevel(level = logging.INFO)
        name=get_struc_name(C.MLP_config["struc"],'MLP')
        handler = logging.FileHandler('logging/%s_log'%name)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.MLP_config,name,logger)
        main_net.train()
    if which=="RBF":
        logger = logging.getLogger(C.RBF_config["which"])
        logger.setLevel(level = logging.INFO)
        name=str(C.RBF_config["struc"])
        handler = logging.FileHandler('logging/%s_log'%name)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.RBF_config,name,logger)
        main_net.train()
    
    if which=="Pade":
        logger = logging.getLogger(C.Pade_config["which"])
        logger.setLevel(level = logging.INFO)
        name='_'.join([get_struc_name(C.Pade_config['struc'],'Pade'),str(C.Pade_config["up_order"]),str(C.Pade_config["down_order"]),str(C.Pade_config["flag"])])
        handler = logging.FileHandler('logging/the_ratio_net_%s_log'%name)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.Pade_config,name,logger)
        main_net.train()
  
