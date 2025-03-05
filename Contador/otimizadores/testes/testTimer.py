import subprocess
from threading import Timer
import shlex

def run(task, timeout_sec):
    print("task =", task)
    kill = lambda process: process.kill()
    process = subprocess.Popen(shlex.split(task),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    timer = Timer(timeout_sec, kill, [process])
    output = ""
    errors = ""
    timedout = False
    try:
        timer.start()
        output, errors = process.communicate()
    finally:
        timer.cancel()

    return (output, errors)




if __name__ == '__main__':
    import time

    t0 = time.perf_counter()
    o, e = run("python printsecs.py 2", 10)  # process ends normally at 1 second
    t1 = time.perf_counter()
    print("time =", t1-t0)
    print("o =", o)
    print("e =", e)
    t0 = time.perf_counter()
    o, e = run("python printsecs.py 11", 10)  # process ends normally at 1 second
    t1 = time.perf_counter()
    print("time =", t1-t0)
    print("o =", o)
    print("e =", e)
    