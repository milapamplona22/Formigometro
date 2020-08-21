#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import yaml
import countline as cl
import argparse
import sys
import cv2

def splitCrossingAndNon(cline, data, key):
    crossed = []
    noncrossing = []
    for trk in data:
        cp1, cm1 = cline.update([trk[key][0], trk[key][-1]])
        ks = trk.keys()
        if (cp1 != 0 or cm1 != 0):
            cross = {}
            for k in ks:
                cross[k] = trk[k]
            if (cp1 != 0 and cm1 == 0):
                cross['direction'] = 1
            elif (cp1 == 0 and cm1 != 0):
                cross['direction'] = -1
            elif (cp1 != 0 and cm1 != 0):
                cross['direction'] = 2
            crossed.append(cross)
        else:
            ncross = {}
            for k in ks:
                ncross[k] = trk[k]
            noncrossing.append(ncross)

    return (crossed, noncrossing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-yml',
                        help="opencv yml file containing tracker output\
                        information",
                        required=True)
    parser.add_argument('-cline',
                        help="countline points sequence",
                        nargs='+',
                        required=True)
    parser.add_argument('-show',
                        help="show which side is which",
                        action='store_true',
                        required=False)
    args = parser.parse_args()
    #if (args.only_noncrossing and args.only_crossing):
    #    print "Options --only-crossing and --only-noncrossing are mutually exclusive"
    #    sys.exit()


    # open OpenCV YAML output from e.g. dataRRANSAC
    #track_data = readYML(sys.argv[2]) # the opencv way
    data = open(args.yml).readlines()[2:] # yml way
    data = ''.join(data)
    data = yaml.load(data)
    if (len(data['tracks']) == 0):
        sys.exit("no tracks in input file "+sys.argv[2])
    # add _id to data (in order of appearance in yml file)
    for i in range(len(data['tracks'])):
        data['tracks'][i]['id'] = i


    args.cline = [int(p) for p in args.cline]
    cline = cl.Countline(args.cline)
    
    for trk in data['tracks']:
        cline.update([trk['position'][0], trk['position'][-1]])

    total = cline.getTotal()
    print("  1:", total[0])
    print(" -1:", total[1])

    if (args.show):
        img = np.ones((data['size'][1], data['size'][0], 3), np.uint8) * 255
        cline.plot(img)
        cv2.imshow("dataCountline", img)
        cv2.waitKey(0)

    # data = []
    # if (args.only_crossing or args.only_noncrossing):
    #     if (args.only_crossing):
    #         data, noncross = splitCrossingAndNon(cline, data['tracks'], 'position')
    #         noncross = []
    #     if (args.only_noncrossing):
    #         crossing, data = splitCrossingAndNon(cline, data['tracks'], 'position')
    #         crossing = []

    #print "%YAML:1.0\n---"
    #print "input:", args.yml
    #print "countline_pts:", args.cline
    
