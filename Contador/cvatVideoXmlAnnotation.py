import sys
import numpy as np
import xml.etree.ElementTree as ET

tree = ET.parse(sys.argv[1])
root = tree.getroot()


def getTaskInfos(root) -> dict :
	task = root.find('meta').find('task')
	meta = {}
	meta['videofilename'] = task.find('name').text
	meta['Nframes'] = int(task.find('size').text)
	meta['labels'] = [label.find('name').text for label in task.find('labels')]
	meta['owner'] = {'name': task.find('owner').find('username').text,
					 'email': task.find('owner').find('email').text}
	return meta


def getTracks(root) -> (str, str, np.ndarray, np.ndarray):
	readIdLabel = lambda t: (t.get('id'), t.get('label'))
	readbox = lambda b: [float(x) for x in [b.get('frame'), b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')]]
	split = lambda x: (x[:,0], x[:,1:])
	readTrack = lambda t: (*readIdLabel(t), *split(np.vstack(list(map(readbox, t.findall('box'))))))
	# (id, label, frames, bboxes)
	return list(map(readTrack, root.findall('track')))

tracks = getTracks(root)
print(tracks[0])

	
	
'''
# one specific item attribute
print('Item #2 attribute:')
print(root[0][1].attrib)

# all item attributes
print('\nAll attributes:')
for elem in root:
    for subelem in elem:
        print(subelem.attrib)

# one specific item's data
print('\nItem #2 data:')
print(root[0][1].text)

# all items data
print('\nAll item data:')
for elem in root:
    for subelem in elem:
        print(subelem.text)
'''