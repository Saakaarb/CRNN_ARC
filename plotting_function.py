


def plot_solution(constants,label=None):

    t_init=constants['t_data'][0]
    t_end=constants['t_data'][-1]+200
    #TODO problem

    y_init=jnp.array([1.0,0.04,397.0])#constants['c_init']
    #y_init=jnp.array([constants['c_init'][0],constants['c_init'][1],constants['c_init'][2]])#constants['c_init']
    saveat = diffrax.SaveAt(steps=True)
    #saveat = diffrax.SaveAt(ts=constants['t_data'])
    term=diffrax.ODETerm(ode_fn)
    solution=diffrax.diffeqsolve(term,diffrax.Kvaerno5(),t0=t_init,t1=t_end,dt0 = 1.0,max_steps=100000,y0=y_init,saveat=saveat,args=constants,stepsize_controller=diffrax.PIDController(pcoeff=0.3,icoeff=0.4,rtol=1e-8, atol=1e-8,dtmin=None))

    num_times=solution.ts.shape[0]
    #dTdt_hist=0.0
    dTdt_loss=0.0
    # get Q
    pred=[]
    pred_stage1=[]
    pred_stage2=[]
    truth=[]
    time=[]
    Temp=[]
    Temp_truth=[]
    print("Predicting solution for num times:",num_times)
    for i in range(num_times):
        if i%100==0:
            print(i)
        #sol_t=ode_fn_plot(constants['t_data'][i],solution.ys[i,:],constants)
        sol_t=ode_fn_plot(solution.ts[i],solution.ys[i,:],constants)
        dTdt_hist=sol_t[2]
        dTdt_hist_stage1=sol_t[3]
        dTdt_hist_stage2=sol_t[4]
        pred.append(dTdt_hist)
        pred_stage1.append(dTdt_hist_stage1)
        pred_stage2.append(dTdt_hist_stage2)
        #truth.append(constants['dTdt_data'][i])
        #time.append(constants['t_data'][i])
        time.append(solution.ts[i])
        Temp.append(solution.ys[i,-1])
        #Temp_truth.append(constants['T_data'][i])
        #dTdt_loss+=jnp.square(jnp.log10(dTdt_hist)-jnp.log10(constants['dTdt_data'][i]))
    Temp_truth=constants['T_data']
    truth=constants['dTdt_data']
    time_truth=constants['t_data']
    #dTdt_loss=jnp.sqrt(dTdt_loss)/num_times

    # do units conversion 
    Temp_truth=np.array(Temp_truth)-273.15
    truth=np.array(truth)*60
    time_truth=np.array(time_truth)/60

    time=np.array(time)/60
    Temp=np.array(Temp)-273.15
    pred=np.array(pred)*60
    pred_stage1=np.array(pred_stage1)*60
    pred_stage2=np.array(pred_stage2)*60

    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.plot(Temp_truth,truth,label='Experimental Data')
    plt.plot(Temp,pred,label='Total Heat Rate')
    plt.plot(Temp,pred_stage1,label='Heat Rate Stage 1',linestyle='dotted')
    plt.plot(Temp,pred_stage2,label='Heat Rate Stage 2',linestyle='dotted')
    plt.yscale('log')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Heat Rate (C/min)')
    #plt.legend(loc='lower right')
    plt.legend( mode='expand', numpoints=1, ncol=2, fancybox = True,fontsize='small',bbox_to_anchor=(0.0,1.0,1.0,0.3),loc='lower left')
    plt.xlim([Temp_truth[0],Temp_truth[-1]])
    plt.grid()
    plt.ylim((10**(-2),10**(4)))
    plt.savefig('check_rate_'+label+'.png')
    plt.close()

    plt.plot(time_truth,Temp_truth,label='Experimental Data')
    plt.plot(time,Temp,label='Predicted Solution')
    plt.legend()
    plt.xlabel('Time (min)')
    plt.ylabel('Temperature (C)')
    plt.gca().set_xlim(left=time_truth[0])
    plt.grid()
    plt.savefig('check_temp_'+label+'.png')
    plt.close()
  
    temp_pred=np.interp(time_truth,time,Temp)
    err=np.mean(np.abs(temp_pred-Temp_truth))
    print("Error "+label,err)

    input("check")



# construct ODE fn for plot

def ode_fn_plot(t,c,constants):

    # parameters for stage 1
    unscaled_A1=jnp.power(10,unscale_val(constants['A1'],constants['log_min_A'],constants['log_max_A']))
    unscaled_Ea1=jnp.power(10,unscale_val(constants['Ea1'],constants['log_min_Ea'],constants['log_max_Ea']))
    unscaled_h1=jnp.power(10,unscale_val(constants['h1'],constants['log_min_h'],constants['log_max_h']))
    
    # parameters for stage 2
    unscaled_A2=jnp.power(10,unscale_val(constants['A2'],constants['log_min_A'],constants['log_max_A']))
    unscaled_Ea2=jnp.power(10,unscale_val(constants['Ea2'],constants['log_min_Ea'],constants['log_max_Ea']))
    unscaled_h2=jnp.power(10,unscale_val(constants['h2'],constants['log_min_h'],constants['log_max_h']))
    
    unscaled_m2=unscale_val(constants['m2'],constants['min_m'],constants['max_m'])
    unscaled_n2=unscale_val(constants['n2'],constants['min_n'],constants['max_n'])
    #--------

    deriv1 = - jnp.power(c[0],constants['n1'])*unscaled_A1*jnp.exp(-unscaled_Ea1/c[-1])
    deriv_1= -jnp.array(1.0)*unscaled_h1*deriv1/(constants['Cp']*constants['mass'])
    deriv_T=deriv_1

    deriv2= jnp.power(c[1],unscaled_n2)*jnp.power(jnp.array(1.0)-c[1],unscaled_m2)*unscaled_A2*jnp.exp(-unscaled_Ea2/c[-1])
    deriv_2= unscaled_h2*deriv2/(constants['Cp']*constants['mass'])
    deriv_T+=deriv_2

    T_inf=jnp.interp(t,constants['t_data'],constants['T_data'])
    h_conv=0.94115*(jnp.abs(T_inf-c[-1])/0.07)**0.35
    Qdiss=constants['Acell']*(h_conv*(T_inf-c[-1])+constants['eps']*constants['sigma']*(jnp.power(T_inf,4)-jnp.power(c[-1],4)))/(constants['Cp']*constants['mass']) 
    deriv_T+=Qdiss
    

    return jnp.stack([deriv1,deriv2,deriv_T,deriv_1,deriv_2])
