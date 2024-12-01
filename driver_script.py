# Main driver script

import numpy as np
import math
from matplotlib import pyplot as plt
import time
import jax
import equinox as eqx
import diffrax
import jax.numpy as jnp
import optax
from sklearn.linear_model import LinearRegression
from itertools import permutations
from crnn_training_funcs import get_linear_estimate,stage,main,unscale_val
#This line is very important, as computation can fail at 32 bit 
jax.config.update("jax_enable_x64", True)

# preprocess ARC data before fitting    
def preprocess_data(time_arc,Temperature_all,Q_exo_all,fit_start_temp):

    for i in range(Temperature_all.shape[0]):
    
        if Temperature_all[i]> fit_start_temp:
            arg_break=i
            break
    time_arc=time_arc[arg_break:]
    Temperature_all=Temperature_all[arg_break:]
    Q_exo_all=Q_exo_all[arg_break:]
    
    # snip data to end Temperature (max)
    
    max_Temp_arg=np.argmax(Temperature_all)
    
    time_arc=time_arc[:max_Temp_arg+1]
    Temperature_all=Temperature_all[:max_Temp_arg+1]
    Q_exo_all=Q_exo_all[:max_Temp_arg+1]
    
    Temp_arc=Temperature_all
    
    # set start time =0
    time_arc=time_arc-time_arc[0]
   
    return time_arc,Temperature_all,Temp_arc,Q_exo_all


if __name__=="__main__":

    #constants
    Cp=jnp.array(859.0)
    Acell=jnp.array(4.618E-3)
    mass=jnp.array(0.066)
    eps=jnp.array(0.8)
    sigma=jnp.array(5.67037442e-8)
    kb=jnp.array(1.380649E-23)


    num_stages=2
    stages_types=['kinetic','all']

    # Initial conditions
    fit_start_temp=397.0
    
    data=np.genfromtxt('data_file.csv',delimiter=',')
    time_arc=data[:,0]
    Temperature_all=data[:,1]
    Q_exo_all=data[:,2]

    #--------
    #  
    n_iters=10000
    # guessed initial conditions of m,n
    m_val=5.0
    n_val=0.0

    #--------
    # write list of variables that should be trained. These may or may not be all possible variables

    trainable_variable_names=['A1','Ea1','h1','A2','Ea2','h2','m2']

    #--------

    # all losses list
    all_losses=[]

    # all params list
    all_params=[]

    time_arc,Temperature_all,Temp_arc,Q_exo_all=preprocess_data(time_arc,Temperature_all,Q_exo_all,fit_start_temp)
    t_init=jnp.array(time_arc[0])

    # linear estimate of fit to get initial guess
    linear_estimate=get_linear_estimate(time_arc,Temperature_all,Q_exo_all,mass,Cp,kb)

    #------------------------------------------
    # create a list of all stage objects
    # stages list

    stages_list=[]
    for i,stagename in enumerate(stages_types):

        stage_obj=stage(stagename,linear_estimate[i],m_val,n_val)
        stages_list.append(stage_obj)

    #setup initial conditions tensor
    #---------------------------
    init_cond=[]
    for stage_obj in stages_list:
        init_cond.append(stage_obj.init_conc)

    init_cond.append(Temperature_all[0])

    #---------------------------
    #collect indices of differentiable params
    diff_list=[]
    
    # setup constants dictionary
    # experimental data
    constants={'Acell':Acell,'mass':mass,'Cp':Cp,'eps':eps,'sigma':sigma}
    constants['t_data']=jnp.array(time_arc)
    constants['T_data']=jnp.array(Temperature_all)
    constants['dTdt_data']=jnp.array(Q_exo_all) 
    constants['c_init']= init_cond
    constants['num_stages'] = num_stages 
    #add scaling values in
    constants['log_max_A']=stage_obj.log_max_A
    constants['log_min_A']=stage_obj.log_min_A
    constants['log_max_Ea']=stage_obj.log_max_Ea
    constants['log_min_Ea']=stage_obj.log_min_Ea
    constants['log_max_h']=stage_obj.log_max_h
    constants['log_min_h']=stage_obj.log_min_h
    constants['min_m']=stage_obj.min_m
    constants['max_m']=stage_obj.max_m
    constants['min_n']=stage_obj.min_n
    constants['max_n']=stage_obj.max_n

    # setup all variables dictionary. These are all variables that can be trained
    all_vars={}

    for i_stage,stage_obj in enumerate(stages_list):
        stage_no=str(i_stage+1)

        all_vars['A'+stage_no]=stage_obj.A

        all_vars['Ea'+stage_no]=stage_obj.Ea
        all_vars['h'+stage_no]=stage_obj.h
        all_vars['m'+stage_no]=stage_obj.m
        all_vars['n'+stage_no]=stage_obj.n


    loss_val,trained_vars=main(constants,all_vars,trainable_variable_names,n_iters)

    # trained vars are scaled, they can be unscaled using the unscale_val function

    # all values aside from m,n would have to be raised to the power of 10 


