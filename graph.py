import numpy as np
from os import listdir, path
import pandas as pd

def check_params(address,parameters):
    simulations = [name for name in listdir(address)]
    keys = [key for key in parameters.keys()]
    file ='log.txt'
    for key,data in parameters.items():
        if len(data) < len(simulations):
            for x,dir in enumerate(simulations):
                loc = path.join(address,dir,file)
                f = open(loc, mode='r')
                sim_keys = []
                for i, line in enumerate(f):
                    line = line.rstrip('\n')
                    vals = line.split(' : ')
                    if len(vals) > 1:
                        param = vals[0]
                        sim_keys.append(param)
                if len(sim_keys) < len(keys):
                    for k in keys:
                        if k not in sim_keys:
                            parameters[k].insert(x,'-')
    return parameters

def log_to_dict(address):
    parameters = {}
    for dir in listdir(address):
        if dir[:3] == 'log':
            loc = path.join(address, dir)
            f = open(loc, mode='r')
            for i, line in enumerate(f):
                line = line.rstrip('\n')
                vals = line.split(':')
                if len(vals) > 1:
                    param = vals[0]
                    if param[:4] == 'Last':
                        break
                    param = param[0:len(param) - 1]
                    val = vals[1]
                    val = val[1:len(val)]
                    try:
                        val = float(val)
                    except:
                        val = str(val)
                    parameters.update({param: val})
    return parameters


def load_results(file, select,pick=None, save_csv=True,latex=False):
    dic = {'sim_id':[]}
    params = None
    c = 0
    for sim in listdir(file):
        sim_path = path.join(file,sim)
        #print(sim_path)
        s_id = sim_path.split('-')[-1]
        dic['sim_id'].append(s_id)
        if c==0:
            parms = log_to_dict(sim_path)
            params = {key:[] for key in parms}
        parms = log_to_dict(sim_path)
        for key in parms:
            if key in params:
                params[key].append(parms[key])
            else:
                params[key] = [parms[key]]
        c+=1
    params = check_params(file,params)
    params_selected = {key:params[key] for key in select}
    dic = {**dic,**params_selected}
    if pick is not None:
        selection_vec = np.ones_like(dic['sim_id'])
        for key,values in pick.items():
            dic_vals = dic[key]
            vec = np.zeros_like(selection_vec)
            for i,val in enumerate(dic_vals):
                if val in values:
                    vec[i] = 1
            selection_vec*=vec
        selection_vec = np.array(selection_vec,dtype=bool)
        for key in dic.keys():
            dic[key] = np.asarray(dic[key])[selection_vec]
    panda = pd.DataFrame(dic)
    if save_csv:
        if latex:
            test_accs, std = dic['CA'],dic['CA_std']
            latex_accs = []
            for acc, var in zip(test_accs,std):
                var = round(var,2)
                acc = round(acc,2)
                latex_accs.append('{} $\pm$ {}'.format(acc,var))
            dic['Test_acc'] =latex_accs
            panda = pd.DataFrame(dic)
            panda.to_csv('results_bd.csv')
        else:
            panda.to_csv('results_bd.csv')
    return dic


select = ['model','model_sur','bd_model','target_label','L2_reg','cos_reg','root_path','CA','CA STD','ASR','ASR STD']

loc = 'Results/Results1'
load_results(loc,select,save_csv=True,latex=False)