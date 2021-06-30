import sys
import os
import numpy as np
import pandas as pd

import sys
sys.path.append("../")
from Contador.otimizadores.otmMngr import RRANSAC_bayesOpt
from Contador.otimizadores import runTasks
from Contador.otimizadores.MOTmetrics import applyMOT, zeroResult
from Contador.dataRRANSAC.dataRRANSACutils import dataRRANSACtask, framewiseMeans2, readTrackerResults, outfilename
from Contador.cvat.cvatVideoXmlAnnotation import readCvatVideoAnnotationsXml, splitFramewise


if __name__ == '__main__':

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
    timeout = 3000
    import time
    def myTask(fp):
        inputfile, params = fp
        # os nomes não podem tem espaço no endereço
        ytrue = params.pop('ytrue')
        #rransacOutfw = outfilename(os.path.basename(inputfile), '_rransacfw.yml')
        rransacOutfw = os.path.join("/dev/shm", outfilename(os.path.basename(inputfile), '_rransacfw.yml'))
        params["outfw"] = rransacOutfw
        rransacOut = os.path.join(os.getcwd(), outfilename(os.path.basename(inputfile)))
        t0 = time.perf_counter()
        outmsg, errmsg = runTasks.run(
            dataRRANSACtask(inputfile, rransacOut, '../Contador/dataRRANSAC/build/',
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
    