#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


class Countline(object):

    class Segment(object):
        def __init__(self, p1, p2):
            # start p1(x,y), end p2(x,y)
            self.p1 = p1
            self.p2 = p2
            self.A1, self.B1, self.C1 = self.calcAuxParams()
            self.nvec = self.normalVec(self.p1, self.p2)

        def calcAuxParams(self):
            A1 = self.p2[1] - self.p1[1]
            B1 = self.p1[0] - self.p2[0]
            C1 = A1 * self.p1[0] + B1 * self.p1[1]
            return (A1, B1, C1)

        def normalVec(self, p1, p2):
            dx = p2[0]-p1[0]
            dy = p2[1]-p1[1]
            return np.array([0., 0., -dy, dx], np.float32)

        def cross(self, p1, p2):
            # vec has to be a 4x1 vector
            return((p1[0] * p2[1]) - (p1[1] * p2[0]));

        def plot(self, img, color=(184,80,85)):
            cv2.line(img, self.p1, self.p2, color, 2, 8)

        @staticmethod
        def intersect(q1, q2, cl1, cl2, As, Bs, Cs):
            # (q1 q2) - start and end point of an external segment
            # (cl1, cl2) - self.p1 and self.p2
            A2 = q2[1] - q1[1];
            B2 = q1[0] - q2[0];
            C2 = A2 * q1[0] + B2 * q1[1];

            det = As*B2 - A2*Bs;
            crossing = False;

            if ( det != 0 ):
                # check if crossing point occurs inside the limits of the segments
                x = (B2*Cs - Bs*C2)/det;
                y = (As*C2 - A2*Cs)/det;
                if ( ( (cl2[0] <= x and x <= cl1[0]) or (cl1[0] <= x and x <= cl2[0]) ) and
                     ( (q2[0] <= x and x <= q1[0]) or (q1[0] <= x and x <= q2[0]) ) and
                     ( (cl2[1] <= y and y <= cl1[1]) or (cl1[1] <= y and y <= cl2[1]) ) and
                     ( (q2[1] <= y and y <= q1[1]) or (q1[1] <= y and y <= q2[1]) ) ):
                    crossing = True;

            return crossing;

        @classmethod
        def getCrossingDirection(cls, segmentInstance, q1, q2, nvec):
            # this function only makes sense if the given external segment (q1->q2)
            # crosses this segment (nvec), and it's been previously checked with
            #  Countline::Segment::intersect() returned true;

            # calculate normal vector of the given external segment (q1, q2)
            nv = segmentInstance.normalVec(q1, q2)
            direcao = segmentInstance.cross((nvec[2], nvec[3]), (nv[2], nv[3]))
            # get position (direction)
            if (direcao >= 0):
                return 1
            else:
                return -1


    def __init__(self, list_of_pts):
        self.segments = [] # list of segments
        self.counts = [] # list of counts in each direction [(1,-1)]

        def initMe(self, pts):
            for i in range(0, len(pts)-1):
                self.segments.append(self.Segment(pts[i], pts[i+1]))
                self.counts.append((0,0))

        if (type(list_of_pts[0]) == list or type(list_of_pts[0]) == tuple):
            initMe(self, list_of_pts)
                
        elif (type(list_of_pts[0])==int or type(list_of_pts[0])==float
            or type(list_of_pts[0])==double):
            lpts = []
            for i in range(0, len(list_of_pts)-1, 2):
                lpts += [(list_of_pts[i], list_of_pts[i+1])]
            initMe(self, lpts)


    def update(self, points):
        # check if a given vector<Points> points crosses any of the segments of
        # this countline
        count_p1 = 0
        count_m1 = 0
        for i in range(0, len(self.segments)): # for each segment
            # for each pair of successive points of vector<Points> points
            for j in range(0, len(points)-1):
                # j has to be int .size()-1 cannot be size_t (no negatives)
                # check if points[j]->[j+1] crosses segment[j]
                if (self.segments[i].intersect(points[j], points[j+1],
                            self.segments[i].p1, self.segments[i].p2,
                            self.segments[i].A1, self.segments[i].B1,
                            self.segments[i].C1)):
                    # if it does, get the direction and increase the respective counter
                    d = self.segments[i].getCrossingDirection(self.segments[i], points[j],
                                    points[j+1], self.segments[i].nvec)
                    # d can only be 1 or -1 (as returned)
                    if (d == 1):
                        self.counts[i] = (self.counts[i][0]+1, self.counts[i][1])
                        count_p1 += 1
                    elif (d == -1):
                        self.counts[i] = (self.counts[i][0], self.counts[i][1]+1)
                        count_m1 -= 1

        return (count_p1, count_m1)


    def plot(self, img, color=(184,80,85)):
        for s in self.segments:
            s.plot(img, color)

    def getTotal(self):
        total_p1 = 0
        total_m1 = 0
        for i in range(len(self.counts)):
            total_p1 += self.counts[i][0]
            total_m1 += self.counts[i][1]
        return (total_p1, total_m1)
