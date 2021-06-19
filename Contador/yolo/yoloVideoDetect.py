
import cv2
import argparse
import numpy as np
from itertools import starmap
import yaml

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(allclasses[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def predict(net, image, scale, Height, Width, input_size=(416,416), mean=(0,0,0), conf_thresh=0.5, nms_thresh=0.4):
    blob = cv2.dnn.blobFromImage(image, scale, input_size, (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    classes = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_thresh:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                classes.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    if nms_thresh > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        indices = [int(i) for i in indices]
        boxes = [boxes[i] for i in indices]
        confidences = [confidences[i] for i in indices]
        classes = [classes[i] for i in indices]

    return boxes, confidences, classes


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=True,
                    help = 'path to input image')
    ap.add_argument('-c', '--config', required=True,
                    help = 'path to yolo config file')
    ap.add_argument('-w', '--weights', required=True,
                    help = 'path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,
                    help = 'path to text file containing class names')
    ap.add_argument('--yml', required=True,
                    help = 'path to output yml file')
    ap.add_argument('-s', '--show', required=False,
                    action='store_true',
                    help = 'show detections on specified video')
    ap.add_argument('--gpu', required=False,
                    action='store_true',
                    help = 'use gpu')
    args = ap.parse_args()


    # abrir o vídeo
    video = cv2.VideoCapture(args.video)
    Height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    Width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

    # ler quais são as classes
    allclasses = None
    with open(args.classes, 'r') as f:
        allclasses = [line.strip() for line in f.readlines()]
    # uma cor para cada classe
    COLORS = np.random.uniform(0, 255, size=(len(allclasses), 3))

    # carregar rede
    net = cv2.dnn.readNet(args.weights, args.config)
    if args.gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    scale = 0.00392

    #def opencv_matrix_representer(dumper, mat):
    #    mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    #    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)
    #yaml.add_representer(np.ndarray, opencv_matrix_representer)

    yml = open(args.yml, 'w')
    yml.write("%YAML:1.0")
    #yaml.dump({"a matrix": np.zeros((10,10)), "another_one": np.zeros((2,4))}, f)
    data = {""}

    blobs = []
    fcount = 0
    while(True):
        # Capture frame-by-frame
        ret, image = video.read()
        if not ret:
            break

        boxes, confidences, classes = predict (net, image, scale, Height, Width, (416,416), (0,0,0))
        draw_predictions = lambda classs, conf, box: draw_prediction_box(image, classs,
            conf, round(box[0]), round(box[1]), round(box[0]+box[2]), round(box[1]+box[3]))
        list(starmap(draw_predictions, zip(classes, confidences, boxes)))

        #for i, c in enumerate(classes):
        #    print(i, allclasses[c], confidences[c])
        if len(boxes) > 0:
            blobs += [{'frame': fcount, 'centroids':[(int(round(box[0])), int(round(box[1]))) for box in boxes]}]

        fcount += 1
        if (args.show):
            cv2.imshow("object detection", image)
            k = cv2.waitKey(10)
            if k == 27:
                break

    video.release()
    cv2.destroyAllWindows()

    outyml = open(args.yml, "w")
    outyml.write("%YAML:1.0\n")
    outyml.write("---\n")
    outyml.write("size: [ %d, %d ]\n"%(Width, Height))
    outyml.write("features:\n")
    for elems in blobs:
        outyml.write("   - { frame:%d, centroids:[ "%(elems["frame"]))
        for i, center in enumerate(elems["centroids"]):
            outyml.write("[ %d, %d ]"%(center[0], center[1]))
            if i < len(elems["centroids"])-1:
                outyml.write(", ")
            else:
                outyml.write(" ")
        outyml.write("] }\n")