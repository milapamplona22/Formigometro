import re
import numpy as np
from multiprocessing import Pool, freeze_support
import pandas as pd


class OptMngr(object):
    # param_bounds -> format depend on derived type (this is a base class)
    # inputfile -> .csv | .xlsx file (each line is:  input_file0; ground truth values columns;)
    # metrics   -> list of metric functions
    # loss      -> loss function on a single input of a iteration
    # reduceLoss -> function that calc loss and sums losses of inputs in the same iteration
    # cpu -> number of processes in multiprocessing
    def __init__(self,
        params:dict,
        task:callable,
        loss:callable,
        metrics:list,
        inputs:list,
        ytrue:np.ndarray,
        cpus:int,
        saveas:str=None):
        checkPair = lambda x: (isinstance(x, tuple) and len(x)==2 and (x[0] <= x[1]))
        assert(all([checkPair(x) for k, x in params.items()]))
        self.params = params
        self.task = task
        self.loss = loss
        self.ytrue = ytrue
        self.inputs = inputs
        self.saveas = saveas
        self.metrics = metrics
        # multiprocessing
        freeze_support()
        self.cpus = np.min([cpus, len(self.inputs)])
        self.pool = Pool(processes=cpus)
        self.history = {'loss':[],
                        'metrics':[],
                        'metrics_names': [f.__name__ for f in metrics],
                        'params': [],
                        'params_names': list(params.keys())}

    def Round(self, **kwargs):
        self.results = self.pool.map(self.task, [(x, dict(kwargs, **{'ytrue':self.ytrue[i]})) for i, x in enumerate(self.inputs)])
        loss_i = self.loss(self.ytrue, self.results)
        metrics_i = [m(self.ytrue, self.results) for m in self.metrics]
        self.history['loss'].append(loss_i)
        self.history['metrics'].append(metrics_i)
        self.history['params'].append([kwargs[x] for x in kwargs])
        if self.history['params_names'] == []:
            self.history['params_names'] = list(kwargs.keys())
        if self.saveas is not None:
            print("ToSave")
            print(self.inputs)
            print(self.history)
            np.save(self.saveas, (self.history, self.inputs))
        return loss_i


from bayes_opt import BayesianOptimization

class RRANSAC_bayesOpt(OptMngr):
    def __init__(self, params:dict, task:callable, loss:callable,
        metrics:list, inputs:list, ytrue:np.ndarray, cpus:int, saveas):

        OptMngr.__init__(self, params, task, loss, metrics, inputs, 
            ytrue, cpus, saveas)

        # bayesian optimization
        self.bo = BayesianOptimization(self.Round, self.params)




if __name__ == '__main__':
    import sys
    import os
    import runTasks
    from MOTmetrics import applyMOT, zeroResult
    sys.path.append("../")
    sys.path.append("../dataRRANSAC")
    from dataRRANSACutils import *
    from cvatVideoXmlAnnotation import readCvatVideoAnnotationsXml, splitFramewise


    data = pd.read_csv(sys.argv[1], delimiter=";", skipinitialspace=True)
    print(data.iloc[0,0])
    print(data.iloc[0,1])

    # change parameter bounds carefully
    # higher tauR are dangerous
    params = {#'M':       (30,   30), # M: Number of stored models
              'U':       (1,    20), # U: Merge threshold
              'tau_rho': (0,     2), # tau_rho: Good model threshold
              'tau_T':   (2,    20), # tau_T: Minimum number of time steps needed for a good model
              #'Nw':      (100, 100), # Nw: Measurement window size
              #'ell':     (50,   50), # ell: Number of RANSAC iterations
              'tauR':    (1,    15), # tauR: Inlier threshold (multiplier)
              'Qm':      (0.5,  15), # Qm
              'Rmx':     (0.5,  15), # Rmx
              'Rmy':     (0.5,  15)} # Rmy

    mae = lambda yhat, y: (np.sum(np.abs(y-yhat))/len(yhat))
    rmse = lambda yhat, y: npqrt(np.sum((y-yhat)**2)/len(yhat))
    mape = lambda yhat, y: np.sum(np.abs((y-yhat)/yhat))/len(yhat)
    rho = lambda yhat, y: scipy.stats.pearsonr(yhat, y)[0]


    toDf = lambda x, cols: pd.DataFrame(x, columns=cols)
    zero_result = lambda: toDf(*zeroResult())
    timeout = 300
    import time
    def myTask(fp):
        inputfile, params = fp
        ytrue = params.pop('ytrue')
        rransacOutfw = os.path.join(os.getcwd(), outfilename(os.path.basename(inputfile), '_rransacfw.yml'))
        params["outfw"] = rransacOutfw
        rransacOut = os.path.join(os.getcwd(), outfilename(os.path.basename(inputfile)))
        # no teste o output já existe
        t0 = time.perf_counter()
        outmsg, errmsg = runTasks.run(
            dataRRANSACtask(inputfile, rransacOut, '../dataRRANSAC/build/',
                {**params, **{'M':30, 'Nw':100, 'ell':50}}),
            timeout)
        t1 = time.perf_counter()
        timedOut = True if outmsg == b'' else False
        print("out msg:", outmsg)
        print("err msg:", errmsg if not timedOut else "timedOut")
        print(f"time: {t1-t0}s")

        if (timedOut):
            return zero_result()

        rransac = framewiseMeans2(readTrackerResults(rransacOutfw))

        if len(rransac['frames']) == 0:
            return zero_result()

        motresults = applyMOT(ytrue[3], ytrue[2], ytrue[0],
            rransac['positions'], rransac['frames'], rransac['ids'])
        print("motresults =\n", motresults)
        return motresults


    motm = lambda yhat, y: y[2]

    def motloss(ytrue, results):
        return np.mean([(r['idf1']).values[0] for r in results])


    ospa = lambda yhat, rransac: ospa(yhat, rransac['tracks'])
    mota = lambda y, results: [(r['mota']).values[0] for r in results]
    motp = lambda y, results: [(r['motp']).values[0] for r in results]

    opt = RRANSAC_bayesOpt(
        params=params,
        task=myTask,
        inputs=data['video'],
        ytrue=list(map(lambda f: splitFramewise(readCvatVideoAnnotationsXml(f)[1]), data.iloc[:,1])),
        loss=motloss,            
        metrics=[mota, motp],
        cpus=4,
        saveas="rransacBayesianOpt.npy")

    opt.bo.maximize(init_points=2, n_iter=20, acq='ei', kappa='kappa')

    #for f in data['filename']:
    #    print(dataRRANSACtask(f, f+"_rransac.yml", 'build', {'M': 50, 'U':10, 'tau_rho': 0.2, 'tau_T': 5, 'Nw': 100, 'ell':100, 'tauR': 1, 'Qm': 5, 'Rmx': 10, 'Rmy':10}))
    