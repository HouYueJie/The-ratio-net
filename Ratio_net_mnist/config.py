
RBF_config={
        'which':'RBF',
        'var_name':'real',
        "hidden_nodes":128,#RBF node
        "struc":128,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }
        
MLP_config={
        'which':'MLP',
        'struc':[[32,'tanh'],[32,'tanh']],#'tanh','relu','sigmoid','swish'
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }

Pade_config={
        'which':'Pade',
        'struc':[[8],[8]],#[[16]],[[16],[16]],[64],[128]......
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        "up_order":3, #2,3,4
        "down_order":3,#2,3,4
        "flag":'fc',#"no_fc","fc"
        }

train_config={
        'CKPT':'ckpt',
        "BATCHSIZE":512,
        "EPOCH":50,
        'STEP_EACH_ITER':600,
        'STEP_SHOW_train':1,
        'STEP_SHOW_test':1,
        'EPOCH_SAVE':10,
        "LEARNING_RATE":0.0001,
        "bound_weight":1,
        "step_unbound":5,
        "decay":False,
        "test_line":False,
        "is_plot":True
}
