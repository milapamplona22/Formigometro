#! /usr/bin/python
import numpy as np
import cv2
import sys
import matplotlib.pylab as plt


def readTracks(file_storage):
    file_node = file_storage.getNode("tracks")
    tracks = []
    for i in range(0, file_node.size()):
        frames_node = file_node.at(i).getNode("frames")
        frames = []
        for f in range(0, frames_node.size()):
            frames.append(frames_node.at(f).real())
        position = []
        position_node = file_node.at(i).getNode("position")
        for p in range(0, position_node.size()):
            pos = []
            for pp in range(0, position_node.at(p).size()):
                pos += [position_node.at(p).at(pp).real()]
            position.append(tuple(pos))
        tracks.append((frames, position))
    return tracks


def plotTracks(Tracks):
    plt.figure()
    for t in Tracks:
        positions = np.asarray(t[1])
        plt.plot(positions[:,0], positions[:,1], '-')
        print positions[-3:,:]

if __name__ == '__main__':
    FS = cv2.FileStorage(sys.argv[1],cv2.FILE_STORAGE_READ)
    tracks = readTracks(FS)
    for i in range(0, len(tracks)):
        print "Track %d:"%(i)
        print "  frames (%d)"%(len(tracks[i][0]))
        print "  position (%d)"%(len(tracks[i][1]))
    FS.release()

    plotTracks(tracks)
    plt.show()
