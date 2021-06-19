import sys
import numpy as np
import xml.etree.ElementTree as ET



def getTaskInfos(root) -> dict :
    meta = root.find('meta')
    task = meta.find('task')
    infos = {}
    infos['task'] = task.find('name').text
    infos['video'] = meta.find('source').text
    infos['Nframes'] = int(task.find('size').text)
    infos['labels'] = [label.find('name').text for label in task.find('labels')]
    infos['owner'] = {'name': task.find('owner').find('username').text,
                     'email': task.find('owner').find('email').text}
    return infos


def getTracks(root) -> [(str, str, np.ndarray, np.ndarray)] :
    readbox = lambda b: [float(x) for x in [b.get('frame'), b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')]]
    ''' get data as list of tracks, as originally in xml '''
    tracks = []
    for t in root.findall('track'):
        tid = t.get('id')
        tlabel = t.get('label')
        frames, bboxes = [], []
        for b in t.findall('box'):
            b = readbox(b)
            frames += [b[0]]
            bboxes += [b[1:]]
        tracks += [(tid, tlabel, np.asarray(frames), np.asarray(bboxes))]
    return tracks
    #readIdLabel = lambda t: (t.get('id'), t.get('label'))
    #readbox = lambda b: [float(x) for x in [b.get('frame'), b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')]]
    #split = lambda x: (x[:,0], x[:,1:])
    #readTrack = lambda t: (*readIdLabel(t), *split(np.vstack(list(map(readbox, t.findall('box'))))))
    # (id, label, frames, bboxes)
    #return list(map(readTrack, root.findall('track')))


def getFramewise(tracks:list) -> [(int, np.ndarray, np.ndarray)] :
    ''' get data as list of frames '''
    startframe = int(np.min([trk[2].min() for trk in tracks]))
    endframe = int(np.max([trk[2].max() for trk in tracks]))
    data = []
    for frame in range(startframe, endframe+1):
        fdata = [frame, [], []]
        ffound = False
        for trk in tracks:
            if frame in trk[2]:
                ffound = True
                fdata[1].append(trk[0])
                frameid = np.where(trk[2] == frame)
                fdata[2].append(trk[3][frameid][0])
        if ffound:
            tuplefdata = lambda x: (x[0], np.array(x[1]), np.array(x[2]))
            data.append(tuplefdata(fdata))
    return data

from functools import reduce
from operator import add
def splitFramewise(tracks:list) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray) :
    ids, labels, frames, positions = [], [], [], []
    for frame in np.unique(sorted(np.hstack([t[2] for t in tracks]))):
        frames += [frame]
        ids += [[]]
        labels += [[]]
        positions += [[]]
        for trk in tracks:
            if frame in trk[2]:
                ids[-1] += [trk[0]]
                labels[-1] += [trk[1]]
                frameid = np.where(trk[2] == frame)
                positions[-1] += [(trk[3][frameid][0])]
    return np.asarray(ids), np.asarray(labels), np.asarray(frames), np.asarray(positions)


def splitFw(data):
    #for i in range(len(data)):
    #    print(len(data[i][2][0][0]))
    '''
    frames = np.zeros(len(data))
    ids, pos = [], []
    for f in range(len(data)):
        frames[f] = data[f][0]
        ids += [data[f][1]]
        pos += [data[f][2]]
        #print(frames[f], ids[-1], len(pos[-1]))
        print(pos[-1])
    return frames, np.array(ids, dtype=object), np.array(pos, dtype=object) 
    '''


def readCvatVideoAnnotationsXml(filename) -> (dict, list) :
    tree = ET.parse(filename)
    root = tree.getroot()
    task = getTaskInfos(root)
    tracks = getTracks(root)
    return task, tracks




if __name__ == '__main__':
    task, tracks = readCvatVideoAnnotationsXml(sys.argv[1])
    print("tracks:", len(tracks))
    for t in tracks[:5]:
        print(f"id: {t[0]}, label: {t[1]}, frames: {t[2].shape}, positions: {t[3].shape}")
    print('...\n')
    print(f"tracks[0] = id: {tracks[0][0]}, label: {tracks[0][1]}")
    for i in range(5):
        print(f"frame: {tracks[0][2][i]}, position: {tracks[0][3][i]}")
    print('...\n')

    print("Framewise:")
    framewise = getFramewise(tracks)
    print("frames:", len(framewise))
    for f in framewise[:5]:
        print(f"frame: {f[0]}, ids: {f[1]}, positions: {f[2]}")
    #import matplotlib.pylab as plt
    #n = [len(x[1]) for x in framewise]
    #frames = [x[0] for x in framewise]
    #plt.figure(figsize=(16,3))
    #plt.bar(frames, n)
    #plt.tight_layout()
    #plt.show()


    ids, labels, frames, positions = splitFramewise(tracks)
    print("ids:", len(ids))
    print("labels:", len(labels))
    print("frames:", len(frames))
    print("positions:", len(positions))
    print(f"\nframe: {frames[0]}")
    print(f"ids: {ids[0]}")
    print(f"positions: {positions[0]}")

    '''
    list of tracks -> list of frames:
                      [frame_i] [ids at frame_i] [pos of ids at frame_i]
    '''
    '''
    frames, ids, positions = splitFw(getFramewise(tracks))
    print("frames:", len(frames))
    print("ids:", len(ids))
    print("positions:", len(positions))
    #for i, p in enumerate(positions):
    #    print(i, p)
    '''