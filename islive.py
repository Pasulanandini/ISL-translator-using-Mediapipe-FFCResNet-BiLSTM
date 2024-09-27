from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from rnn_utils import get_network_wide
from collections import deque
from sklearn.model_selection import train_test_split
import tflearn
import os
import sys
import tensorflow as tf
import cv2
from os.path import join, exists
import cropping2 as hs
import numpy as np
from os import startfile
import shutil
hc = []

def convert():
    gesture_folder = r"C:\Users\SANDHYA\Downloads\livefiles\livefiles\isllive0"
    target_folder = r"C:\Users\SANDHYA\Downloads\livefiles\livefiles\islliveframes"
    #rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not exists(majorData):
        os.makedirs(majorData)
    gesture_folder = os.path.abspath(gesture_folder)
    os.chdir(gesture_folder)
    frames = os.listdir(os.getcwd())
 
    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory containing frames: %s\n" % (majorData))

    lastFrame = None
    os.chdir(majorData)
    count = 0
    #print(name)
    # assumption only first 200 frames are important
    while count<201:
        framen = r"C:\Users\SANDHYA\Downloads\livefiles\isllive0/original_frame"+str(count)+".jpeg"
        if not exists(framen):
            break
        frame=cv2.imread(framen)
        framename = "mp_frame_" + str(count) + ".jpeg"
        hc.append([join(majorData, framename)])

        if not os.path.exists(framename):
            #frame=cv2.resize(frame,(540,960))
            frame = hs.cropping2(frame)
            frame=cv2.resize(frame,(960,540))
            lastFrame = frame
            
            cv2.imwrite(framename, frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1

    while count<201:
        framename="mp_frame"+str(count)+".jpeg"
        hc.append([framename])
        if not os.path.exists(framename):
            cv2.imwrite(framename,lastFrame)
        count+=1
        os.chdir(majorData)


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(frames, input_height=224, input_width=224, input_mean=0, input_std=255):
    input_name = "file_reader"
    frames = [(tf.read_file(frame, input_name), frame) for frame in frames]
    decoded_frames = []
    for frame in frames:
        file_name = frame[1]
        file_reader = frame[0]
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
        decoded_frames.append(image_reader)
    float_caster = [tf.cast(image_reader, tf.float32) for image_reader in decoded_frames]
    float_caster = tf.stack(float_caster)
    resized = tf.image.resize_bilinear(float_caster, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def predict(graph, image_tensor, input_layer, output_layer):
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(
            output_operation.outputs[0],
            {input_operation.outputs[0]: image_tensor}
        )
    results = np.squeeze(results)
    return results

def predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size):
    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 255
    batch_size = batch_size
    graph = load_graph(model_file)

    labels_in_dir = os.listdir(frames_folder)
    each=[]
    each = [each for each in labels_in_dir]
    #print(len(each))
    predictions=[]
    for i in range(0, len(each), batch_size):
        batch = each[i:i + batch_size]
        batch = [os.path.join(frames_folder, frame) for frame in batch]
        #print(len(batch))
        try:
            
            frames_tensors = read_tensor_from_image_file(batch, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
            pred = predict(graph, frames_tensors, input_layer, output_layer)
            predictions.extend(pred)
        except KeyboardInterrupt:
            print("You quit with ctrl+c")
            sys.exit()

        except Exception as e:
            print("Error making prediction: %s" % (e))
            x = input("\nDo You Want to continue on other samples: y/n")
            if x.lower() == 'y':
                continue
            else:
                sys.exit()
    return predictions

def get_data(frames):
    """Get the data from our saved predictions or pooled features."""
    temp_list = deque()
    X=[]
    for i, frame in enumerate(frames):

        features = frame[0]
        #actual = frame[1].lower()

        # frameCount = frame[2]

        # Convert our labels into binary.
        #actual = labels[actual]

        # Add to the queue.
        if len(temp_list) == 201 - 1:
            temp_list.append(features)
            flat = list(temp_list)
            X.append(flat)
            #y.append(actual)
            temp_list.clear()
        else:
            temp_list.append(features)
            continue
    return X
    
def load_labels(label_file):
    label = {}
    count = 0
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label[l.strip()] = count
        count += 1
    return label
framename1=r"C:\Users\SANDHYA\Downloads\livefiles\livefiles\isllive0/"
framename=r"C:\Users\SANDHYA\Downloads\livefiles\livefiles/islliveframes/"
shutil.rmtree(framename1)
shutil.rmtree(framename)
if not exists(framename1):
    os.makedirs(framename1)
if not exists(framename):
    os.makedirs(framename)
cap=cv2.VideoCapture(0)
count=-5
print("record ready")
while count<201:
    ret,img=cap.read()
    print(count)
    if img is None:
        break
    if count>=0:
        framen1="/isllive0/original_frame"+str(count)+".jpeg"
        cv2.imshow('abc',cv2.resize(img,(720,405)))
        cv2.imwrite(framen1,img)
        key = cv2.waitKey(5) & 0xFF
        if key == ord("v"):
            break
    count=count+1
    #cv2.imwrite(framen, frame)
cap.release()
cv2.destroyAllWindows()


convert()
predc=predict_on_frames(r"C:\Users\SANDHYA\Downloads\livefiles\livefiles/islliveframes",r"C:\Users\SANDHYA\Downloads\livefiles\livefiles/frozen_model.pb",'IteratorGetNext','predict/Softmax',201)
model_file="non_pool.model"
size_of_each_frame = 76
net = get_network_wide(201, size_of_each_frame, 76)
model = tflearn.DNN(net, tensorboard_verbose=0)
try:
    model.load('checkpoints/' + model_file)
    print("\nModel Exists! Loading it")
    print("Model Loaded")
except Exception:
    print("\nNo previous checkpoints of %s exist" % (model_file))
    print("Exiting..")
    sys.exit()
label=load_labels(r"C:\Users\SANDHYA\Downloads\livefiles\livefiles/retrained_labels.txt")
#print(label)
predictions = model.predict([predc])
#print(predictions)
predictions = np.array([np.argmax(pred) for pred in predictions])
#print(predictions)
print(list(label.keys())[predictions[0]])
