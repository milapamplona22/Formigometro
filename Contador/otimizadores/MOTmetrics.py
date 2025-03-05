import numpy as np
import motmetrics as motm

''' Distance functions
'''
def euclidian4ptlists(gt, obs):
    # rows = gt, cols = observed
    #print("gt  =", gt)
    #print("obs =", obs)
    dists = np.zeros((len(gt), len(obs)), np.float64)
    for i, g in enumerate(gt):
        for j, o in enumerate(obs):
            dists[i, j] = np.sqrt((g[0]-o[0])**2 + (g[1]-o[1])**2)
    return dists


def zeroResult():
    cols = ('idf1', 'idp', 'idr', 'recall', 'precision', 'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_transfer', 'num_ascend', 'num_migrate') 
    return ([[0]*len(cols)], cols)


def applyMOT(gtPositions, gtFrames, gtIds, 
    trkPositions, trkFrames, trkIds, distfunc=euclidian4ptlists):
    #time = np.arange(
    #    np.min([gtFrames.min(), trkFrames.min()]),
    #    np.max([gtFrames.max(), trkFrames.max()]))
    time = np.unique(np.sort(np.hstack([gtFrames, trkFrames])))
    # Initialize py-motmetrics accumulator
    acc = motm.MOTAccumulator(auto_id=True)

    for t in time:
        tidGt, tidTrk = None, None
        gtpos, trkpos = [], []
        gtids, trkids = [], []
        #print("frame", t)
        if t in gtFrames:
            tidGt = np.where(gtFrames==t)[0][0]
            #print("tidGt", tidGt)
            gtids = gtIds[tidGt]
            #print("gtids", gtids)

            
        if t in trkFrames:
            tidTrk = np.where(trkFrames==t)[0][0]
            #print("tidTrk", tidTrk)
            trkids = trkIds[tidTrk]
            #print("trkids", trkids)
        
        if len(gtids) > 0 and len(trkids) > 0:
            dstMat = distfunc(gtPositions[tidGt], trkPositions[tidTrk])
            acc.update(gtids, trkids, dstMat)

    mh = motm.metrics.create()
    summary = mh.compute(acc,metrics=motm.metrics.motchallenge_metrics)
    return summary


''' MOTmetrics (pymotmetrics related stuff)
'''
def MOTmetricsEval(ann_data, tracker_data, distfunc=euclidian4ptlists):
    # *_data [{
    #        ['frames']=[frame numbers]
    #        ['position'] = [(x,y)]]
    #        }] at least
    # begin = np.min([
    #     np.min([map(lambda t: np.min(t['frames']), ann_data)[0]]),
    #     np.min([map(lambda t: np.min(t['frames']), tracker_data)[0]])
    #     ])[0]
    # end = np.max([
    #     np.max([map(lambda t: np.max(t['frames']), ann_data)[0]]),
    #     np.max([map(lambda t: np.max(t['frames']), tracker_data)[0]])
    #     ])[0]

    begin = np.min([
        np.min([*map(lambda t: np.min(t['frames']), ann_data)]),
        np.min([*map(lambda t: np.min(t['frames']), tracker_data)])
        ])
    end = np.max([
        np.max([*map(lambda t: np.max(t['frames']), ann_data)]),
        np.max([*map(lambda t: np.max(t['frames']), tracker_data)])
        ])

    # Initialize py-motmetrics accumulator
    acc = motm.MOTAccumulator(auto_id=True)

    # loop through frame range updating pymotmetrics
    for f in range(begin, end+1):
        
        # get annotations present in frame f
        ann_in_f = []
        ann_in_f_pos = []
        for anntrk in ann_data:
            if f in anntrk['frames']:
                ann_in_f += [anntrk['id']] # indentity index
                ann_in_f_pos += [anntrk['pos'][anntrk['frames'].index(f)]] # it's position

        if (len(ann_in_f) > 0): # only eval if there's gt to that (make it optional)
            # get tracked tracks present in frame f
            trked_in_f = []
            trked_in_f_pos = []
            i = 0
            for trkedtrk in tracker_data:
                if f in trkedtrk['frames']:
                    trked_in_f += [i] # identity index (use order of appearance in yml file),
                    #trked_in_f_pos += [trkedtrk['position'][trkedtrk['frames'].index(f)]] # acha a primeira position para o frame f
                    # acha a última position para o frame f
                    ii = len(trkedtrk['frames']) - 1 - trkedtrk['frames'][::-1].index(f)
                    trked_in_f_pos += [trkedtrk['position'][ii]]
                i += 1

            # calc the distance matrix between objects in this frame
            distMat = distfunc(ann_in_f_pos, trked_in_f_pos)

            # if (len(ann_in_f) > 0 or len(trked_in_f) > 0):
            #     print(f)
            #     print(' gt  :', ann_in_f, ann_in_f_pos)
            #     print(' hypo:', trked_in_f, trked_in_f_pos)
            #     print (distMat)

            acc.update( ann_in_f, trked_in_f, distMat)

    # 3. compute metrics
    mh = motm.metrics.create()
    summary = mh.compute(acc,
        metrics=['num_matches', 'num_detections', 'num_unique_objects', 'num_false_positives', 'num_misses', 'num_switches', 'mota']) 
        #metrics=motm.metrics.motchallenge_metrics)

    return (mh, summary)
