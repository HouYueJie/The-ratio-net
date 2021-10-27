
RBF_config={
        'which':'RBF',
        'var_name':'real',
        "hidden_nodes":64,#RBF node
        "struc":64,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }
        
MLP_config={
        'which':'MLP',
        'struc':[[512,'swish'],[512,'swish']],#'tanh','relu','sigmoid','swish'
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }

Pade_config={
        'which':'Pade',#Pade,MLP
        'struc':[[8],[8]],#'tanh','relu'
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        "up_order":20, #2,3,4
        "down_order":3,#2,3,4
        "flag":'fc',#"no_fc","fc"        
        }
        
train_config={
        'CKPT':'ckpt',
        "BATCHSIZE":512,
        "MAX_ITER":50,
        'STEP_EACH_ITER':600,
        'STEP_SHOW_train':1,
        'STEP_SHOW_test':1,
        'EPOCH_SAVE':10,
        "LEARNING_RATE":0.001,#
        "bound_weight":1,
        "step_unbound":5,
        "decay":False,
        "test_line":False,
        "is_plot":True
}
