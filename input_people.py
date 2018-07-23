import numpy as np
import cv2
import argparse
import os
import tensorflow as tf
import mtcnn_detect
import facenet
import json
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk
from scipy import misc
from random import randint
import math
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("name", help = "name of person need to add")
args = parser.parse_args()

embedding_size = 128
directory = "./dataImg/"+args.name+'/'
model_path = './models/20170512-110547.pb'

im_arr = []

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
image_size = 160

if not os.path.exists(directory):
    os.makedirs(directory)

# mouse callback function
def get_image(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        im_arr.append(img)
        print('Image taken')
def getimage(frame):
    im_arr.append(frame)
    print('Image taken')
labels = []
embs = []
class_names = []
with open('class.txt') as file:
    for l in file.readlines():
        class_names.append(l.replace('\n', ''))
file.close()

print(class_names)
with open('data.txt') as json_file:  
    data = json.load(json_file)
    for p in data['person']:
        embs.append(p['emb'])
        labels.append(p['name'])

if args.name in class_names:
    person_label = class_names.index(args.name)
    print('This person is already in database')
else:
    person_label = len(class_names) 
    file = open('class.txt','w')  
    class_names.append(args.name)
    for name in class_names:
        file.write(name + os.linesep)
    file.close()                 
        
   
# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',get_image)

cap = cv2.VideoCapture(0)
while(1):
    ret, img = cap.read()
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = mtcnn_detect.create_mtcnn(sess, 'models')
cropped_im = []
save_im = []
for i in range(len(im_arr)):
    frame = im_arr[i];
    bounding_boxes, _ = mtcnn_detect.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces == 1:
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(frame.shape)[0:2]

        cropped = []
        scaled = []
        scaled_reshape = []
        bb = np.zeros((nrof_faces,4), dtype=np.int32)

        for i in range(nrof_faces):
            emb_array = np.zeros((1, embedding_size))
            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]
                    
            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
            cropped = cv2.resize(cropped, (image_size,image_size),interpolation=cv2.INTER_CUBIC)
            save_im.append(cropped)
            cropped = prewhiten(cropped)
            cropped = flip(cropped, False)
            cropped_im.append(cropped)
for i in range(len(save_im)):
    cv2.imwrite(directory+str(i)+'.jpg',save_im[i])
print('Extracted: %d' % len(cropped_im))
cropped_im = np.array(cropped_im)
cropped_im.reshape(-1,image_size,image_size,3)

with tf.Graph().as_default():
    with tf.Session() as sess:
        print('Loading feature extraction model')
        facenet.load_model(model_path)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        nrof_images = len(im_arr)
        emb_array = np.zeros((nrof_images, embedding_size))
        
        feed_dict = { images_placeholder:cropped_im, phase_train_placeholder:False }
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        for i in range(len(emb_array)):
            data['person'].append({'name':person_label,'emb':emb_array[i].tolist()})
            labels.append(person_label)
            embs.append(emb_array[i])
        with open('data.txt', 'w') as outfile:
            json.dump(data, outfile)
        print(len(embs))
        print(len(labels))
        X_train, X_test, y_train, y_test = train_test_split(np.array(embs),np.array(labels), test_size=0.33, random_state=42)
        print('Training SVM classifier')
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        accuracy = np.mean(np.equal(best_class_indices, y_test))
        print('Accuracy: %.3f' % accuracy)
        with open('svm_classifier.pkl', 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved svm classifier')

          
