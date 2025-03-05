import yaml
import os
import numpy as np


def readTrackerResults(filename):
    # open OpenCV YAML output from e.g. dataRRANSAC
    #track_data = readYML(sys.argv[2]) # the opencv way
    data = open(filename).readlines()[2:] # yml way
    data = ''.join(data)
    data = yaml.load(data, Loader=yaml.FullLoader)
    # add _id to data (in order of appearance in yml file)
    return data


def dataRRANSACtask(inputfile:str, outputfile:str, path:str, params:dict) -> str:
    required_params = ['M', 'U', 'tau_rho', 'tau_T', 'Nw', 'ell', 'tauR', 'Qm', 'Rmx', 'Rmy']
    assert(all([k in params for k in required_params]))
    params_str = ' '.join([f"-{k} {params[k]}" for k in params.keys()])
    command = f"{os.path.join(path, 'dataRRANSAC')} -data {inputfile} -out {outputfile} "+ params_str
    return command


def framewiseMeans(data):
    selectId = lambda f, i: np.array(data['frames'][f]['id']) == i
    ints = lambda x: list(map(int, x))
    PositionFilter = lambda x: (x[0] >= 0) and (x[0] < data['size'][0]) and (x[1] >= 0) and (x[1] < data['size'][1])
    filterpos = lambda x: list(filter(PositionFilter, x))
    for f in range(len(data['frames'])):
        frame_data = {'frame': [int(x[6:]) for x in data['frames'][f].keys() if x[:5] == 'frame'][0],
            'id': [], 'position': []}
        idpos = [(i, filterpos(np.array(data['frames'][f]['position'])[selectId(f,i)]))
                    for i in ints(np.unique(data['frames'][f]['id']))]
        idpos = list(filter(lambda x: len(x[1]) > 0, idpos))
        idpos = [(i, tuple(np.vstack(pos).mean(axis=0))) for i, pos in idpos]
        for i, pos in idpos:
            frame_data['id'] += [i]
            frame_data['position'] += [pos]
        data['frames'][f] = frame_data
    return data


def framewiseMeans2(data):
    PositionFilter = lambda x: (x[0] >= 0) and (x[0] < data['size'][0]) and (x[1] >= 0) and (x[1] < data['size'][1])
    
    def filterPositions(idsf, positionsf):
        ''' remove ids and positions where position is outside the image size '''
        return list(zip(*[(i, p) for i, p in zip(idsf, positionsf) if PositionFilter(p)])) or ([], [])

    def filterEmptyFrames(frames, ids, positions):
        ''' remove frames ids and positions where positions are empty (from filtering) '''
        return zip(*[(frames[f], ids[f], positions[f]) for f in range(len(frames)) if len(positions[f])>0]) or ([],[],[])

    for f in range(len(data['frames'])):
        data['ids'][f], data['positions'][f] = filterPositions(data['ids'][f], data['positions'][f])
        data['ids'][f] = np.asarray(data['ids'][f], dtype=object)
        data['positions'][f] = np.asarray(data['positions'][f], dtype=object)
        # if id is repeated within frame, average it's position        
        if len(data['positions'][f]) > 0:
            uids, cuids = np.unique(data['ids'][f], return_counts=True)
            if np.sum(cuids > 1) > 1:
                sids = uids[cuids == 1] # ids that appear only once (singles)
                spos = np.vstack([data['positions'][f][data['ids'][f]==u] for u in sids]) if len(sids) > 0 else np.array((0,2))
                rids = uids[cuids > 1] # repeated ids
                rpos = np.asarray([np.vstack(data['positions'][f][data['ids'][f]==r]).mean(axis=0) for r in rids]) if len(rids) > 0 else np.array((0,2), dtype=object)
                data['ids'][f] = np.hstack([sids, rids])
                data['positions'][f] = np.vstack([spos, rpos]) if len(sids) > 0 else rpos
                if len(data['ids'][f]) != len(data['positions'][f]):
                    print("ERRO", f, data['ids'][f].shape, data['positions'][f].shape)
                    assert(len(data['ids'][f]) == len(data['positions'][f]))
        
    data['frames'], data['ids'], data['positions'] = filterEmptyFrames(
        data['frames'], data['ids'], data['positions'])
    data['frames'] = np.asarray(data['frames'])
    data['ids'] = np.asarray(data['ids'], dtype=object)
    data['positions'] = np.asarray(data['positions'], dtype=object)
    return data


def fwSplit(data):
    frames = np.zeros(len(data['frames']))
    ids, pos = [], []
    for f in range(len(data['frames'])):
        frames[f] = data['frames'][f]['frame']
        ids += [data['frames'][f]['id']]
        pos += [data['frames'][f]['position']]
    data['frames'] = frames, np.array(ids, dtype=object), np.array(pos, dtype=object) 
    return data


def outfilename(inputfile:str, suffix="_rransac.yml"):
    return os.path.splitext(inputfile)[0] + suffix


if __name__ == "__main__":
    import sys
    # por enquanto esse tipo de saída é o mais confiável porque as médias de posição
    # para um tracker de mesmo id são realizadas nessa leitura do yaml
    data = framewiseMeans2(readTrackerResults(sys.argv[1]))