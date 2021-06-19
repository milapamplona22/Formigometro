import time

def printSecs(length):
    print(f"sleep for {length}")
    for i in range(length):
        print(i, end='')
        for i in range(10):
            time.sleep(0.1)
            print(".", end='')
    time.sleep(0.999)

if __name__ == '__main__':
    import sys
    printSecs(int(sys.argv[1]))