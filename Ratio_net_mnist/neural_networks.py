import tensorflow.compat.v1 as tf 

#initializer variable
def get_initializer(ini_name):
    if ini_name=="tru_norm":
        weight_initialization =  tf.truncated_normal_initializer(stddev=0.1)
    elif ini_name=="xavier":
        weight_initialization =  tf.contrib.layers.xavier_initializer()
    elif ini_name=="const":
        weight_initialization = tf.constant_initializer(0.1)
    elif ini_name=="uniform":
        weight_initialization =tf.random_uniform_initializer()
    elif ini_name=="scal":
        weight_initialization = tf.variance_scaling_initializer()
    elif ini_name=="orth":
        weight_initialization = tf.orthogonal_initializer()
    else:
        print("error initializer method, please cheack config.py.\
               The initializer funcation choice from ['tru_norm',\
               'xavier','const','uniform','scal','orth']")
        exit()
    return weight_initialization

#setting activate function
def get_activate(act_name):
    if act_name=="sigmoid":
        return tf.nn.sigmoid
    elif act_name=="tanh":
        return tf.nn.tanh
    elif act_name=="relu":
        return tf.nn.relu
    elif act_name=="swish":
        return tf.nn.swish
    else:
        print("error activate funcation method, please cheack config.py.\
               The activate funcation choice from['tanh','relu','sigmoid','swish']")
        exit()

#full connection layer
def full_connect_layer(input_,input_shape,output_shape,var_name,weight_initialization,activate=None,name=None):
    w = tf.get_variable(var_name + 'weight_' + name, 
                        shape=[input_shape, output_shape], 
                        initializer=weight_initialization, 
                        dtype=tf.float32)
    b = tf.get_variable(var_name + 'bias_' + name, 
                        shape=[output_shape], 
                        initializer=weight_initialization, 
                        dtype=tf.float32)
    if activate!=None:
        full_put = activate(tf.add(tf.matmul(input_, w), b))
    else:
        full_put = tf.matmul(input_, w) + b
    return full_put

#pade layer
def pade_build(input_,input_shape,output_shape,var_name,weight_initialization,up_order=2,down_order=2,layer=None,flag=None):
    for u_order in range(up_order):
        if u_order==0:
            up_formula=full_connect_layer(input_,input_shape,output_shape,\
                                          var_name,weight_initialization,name='u_%s_%s_%s'%(layer,u_order,flag))
        else:
            up_formula*=full_connect_layer(input_,input_shape,output_shape,\
                                           var_name,weight_initialization,name='u_%s_%s_%s'%(layer,u_order,flag))
    for d_order in range(down_order):
        if d_order==0:
            down_formula=full_connect_layer(input_,input_shape,output_shape,\
                                          var_name,weight_initialization,name='d_%s_%s_%s'%(layer,d_order,flag))
        else:
            down_formula*=full_connect_layer(input_,input_shape,output_shape,\
                                           var_name,weight_initialization,name='d_%s_%s_%s'%(layer,d_order,flag))

    value  = up_formula/down_formula
    return value
     

#MLP model
class MLP:
    def __init__(self,config,n_input,n_output):
        self.struc=config['struc']
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output=n_output
        self.n_input=n_input
        #####
        print('The MLP model parametets as follows:\n')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        #####
        self.construct_input()
        self.build_value()

    def get_trainable_variables(self):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def construct_input(self):
        print("build placeholder")
        self.input=tf.placeholder(tf.float32, [None, self.n_input])
    def build_value(self):
        print("build MLP network structure")
        for i,stru in enumerate(self.struc):
            this_num,this_act=stru
            activate=get_activate(this_act)
            if i == 0:
                self.layer = full_connect_layer(self.input,self.n_input,\
                                                this_num,self.var_name,\
                                                self.weight_initialization,activate,name=str(i))
            else:
                self.layer = full_connect_layer(self.layer,self.struc[i-1][0],\
                                                this_num,self.var_name,\
                                                self.weight_initialization,activate,name=str(i))
        
        self.value = full_connect_layer(self.layer,self.struc[-1][0],\
                                        self.n_output,self.var_name,\
                                        self.weight_initialization,name=str(len(self.struc)))

#RBF Model
class RBF: 
    def __init__(self,config,n_input,n_output):
        self.n_input=n_input
        self.hidden_nodes=config["hidden_nodes"]
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = n_output
        #######
        print('The RBF model parametets as follows:\n')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()

    def get_trainable_variables(self):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
    def construct_input(self):
        print("build placeholder")
        self.input=tf.placeholder(tf.float32, [None, self.n_input])   
    
    def build_value(self):
        print("build RBF network structure")
        
        self.distance=[]
        self.delta = tf.get_variable(self.var_name+"_delta",
                                     shape      = [self.hidden_nodes],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float32)
        self.delta_2= tf.square(self.delta)

        for i in range(self.hidden_nodes):
            this_center = tf.get_variable(self.var_name + 'center_' + str(i),
                                     shape      = [self.n_input],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float32)

            this_dist=tf.reshape(tf.reduce_sum((self.input - this_center)**2,axis=1),[-1,1])
            self.distance.append(this_dist)

        self.distance_ca=tf.concat(self.distance,axis=1)

        self.out_hidden=tf.exp(-1.0*(self.distance_ca/2*self.delta_2))

        w = tf.get_variable(self.var_name + 'weight' , 
                            shape=[self.hidden_nodes, self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float32)
        b = tf.get_variable(self.var_name + 'bias', 
                            shape=[self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float32)
        self.value=tf.matmul(self.out_hidden,w)+b
        
class Pade: 
    def __init__(self,config,n_input,n_output):
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.struc =  config['struc']
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = n_output
        self.n_input=n_input
        self.up_order=config['up_order']
        self.down_order=config['down_order']
        self.flag=config['flag']
        ######
        print('The Pade model parametets as follows:\n')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()

    def get_trainable_variables(self):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def construct_input(self):
        print("build placeholder")
        self.input=tf.placeholder(tf.float32, [None, self.n_input])   
    def build_value(self):
        print("build network structure")
        for i,stru in enumerate(self.struc):
            this_num=stru[0]
            if self.flag=='no_fc':
                if len(self.struc) ==1:
                    self.value = pade_build(self.input,self.n_input,self.n_output,self.var_name,\
                                            self.weight_initialization,up_order=self.up_order,down_order=self.down_order,layer=0,flag=self.flag)
                else:
                    if i == 0:
                        self.layer = pade_build(self.input,self.n_input,this_num,self.var_name,\
                                                self.weight_initialization,up_order=self.up_order,down_order=self.down_order,layer=i,flag=self.flag)
                    elif i!=len(self.struc)-1:
                        self.layer = pade_build(self.layer,self.struc[i-1][0],this_num,self.var_name,\
                                                self.weight_initialization,up_order=self.up_order,down_order=self.down_order,layer=i,flag=self.flag)
                    else:
                        self.value = pade_build(self.layer,self.struc[i-1][0],self.n_output,self.var_name,\
                                                self.weight_initialization,up_order=self.up_order,down_order=self.down_order,layer=i,flag=self.flag)
            if self.flag=='fc':
                if i == 0:
                    self.layer = pade_build(self.input,self.n_input,this_num,self.var_name,\
                                            self.weight_initialization,up_order=self.up_order,down_order=self.down_order,layer=i,flag=self.flag)
                else:
                    self.layer = pade_build(self.layer,self.struc[i-1][0],this_num,self.var_name,\
                                            self.weight_initialization,up_order=self.up_order,down_order=self.down_order,layer=i,flag=self.flag)
                if i==len(self.struc)-1: 
                    self.value = full_connect_layer(self.layer,self.struc[-1][0],self.n_output,self.var_name,\
                                                    self.weight_initialization,name="fc_"+str(len(self.struc))+str(i))
