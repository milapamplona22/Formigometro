import subprocess
from threading import Timer
import shlex

def run(task, timeout_sec):
    kill = lambda process: process.kill()
    process = subprocess.Popen(shlex.split(task),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    timer = Timer(timeout_sec, kill, [process])
    output = ""
    errors = ""
    try:
        timer.start()
        output, errors = process.communicate()
    finally:
        timer.cancel()

    return (output, errors)


def run_simple(task):
    process = subprocess.Popen(task,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    output, errors = process.communicate()
    return (output, errors)
