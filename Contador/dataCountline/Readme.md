19.06.2021

I think the countline is too complicated for this case. Analyzing the trajectories offline, one by one, and checking at which moments y > ycountline and y < ycountline would be more practical and provide a more interpretable output.
It is quite simple, it could be done from scratch.

---

10/2018
## Glossary:
- countline: set of segments
- segment: unit (section) of the countline
- limit points: (p1,p2) points that define a segment
- external point: (q1,q2) points that will be given to the countline to check whether there was a crossing or not
- trajectory: successive sequence of external points

---

## Usage:
>```
Countline cline({cv::Point(x1,y1), cv::Point(x2,y2), ...})
Countline cline({x1,y1,x2,y2,...}
// the segments will be: p1->p2, p2->p3, ...
// counting at each pair of successive points
cline.update(vector<cv::Point>trajectory)
// counting disregarding u-turns
cline.update({trajectory[0], trajectory[-1]}) // (-1: last element)
```

The testCountline.cpp allows drawing a trajectory with the mouse over a countline and, by pressing 'c', it counts and prints the result.

---

## To Do:
1. Insert int Countline::n (number of segments in the countline)


## Issues:
### 1. When an external point falls on the line
[Currently Counts]. When a trajectory has a point that falls exactly on a segment, for example:
> segment s = [x1=0, y1=120, x2=320, y2=120]
>
> trajectory t = [(100,100), (100,120), (100,130)]

If the trajectory is analyzed in pairs of successive points, the trajectory will cross the line twice: both in the segment [(100,100),(100,120)] and in the following segment [(100,120), (100,130)], since the point (100,120) falls exactly on the countline segment.

Not counting when the external point falls on the line does not solve the issue because, in the same case, it would not produce any count.

Passing only the first and last point of a trajectory to the countline - a method that also resolves half-turn counts - virtually solves the issue (unless the first or last point falls on the segment, but the probability is lower since the countline is usually more central).

### 2. 90° and 180° Lines
When there is a straight line (segment) at multiples of 90° angles - where x1==x2 or y1==y2 - all points on the segment (including those between the limit points) are on integer values. This makes it easier for external points to fall on the countline. Adding a slight tilt to the line (e.g., y1=120, y2=121) makes the intermediate points of the segment floats, thus making it harder for an external point to fall on the countline.

# Log:
- 10/2018 - creation
- 10/2018 - added void Countline::getTotals()
