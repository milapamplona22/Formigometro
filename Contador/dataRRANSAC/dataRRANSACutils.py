import yaml

def readTrackerResults(filename):
    # open OpenCV YAML output from e.g. dataRRANSAC
    #track_data = readYML(sys.argv[2]) # the opencv way
    data = open(filename).readlines()[2:] # yml way
    data = ''.join(data)
    data = yaml.load(data)
    # add _id to data (in order of appearance in yml file)
    return data