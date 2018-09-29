import tkinter 
from tkinter import *
from tkinter import messagebox
import PIL.Image, PIL.ImageTk
import time
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
import sqlite3

embedding_size = 128
model_path = './models/20170512-110547.pb'
import sqlite3

sqlite_file = 'DATAHS.db'
table_name = 'DATAHS'

# Connecting to the database file
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

im_arr = []

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
image_size = 160

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



conn = sqlite3.connect('DATAHS.db')
def query(conn, number):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM DATAHS WHERE field1='"+str(number)+"'")
 
    rows = cur.fetchall()
 
    for row in rows:
        print(row)
    return rows
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.video_source = video_source
        menuBar = Menu(self.window)
        self.window.config(menu=menuBar)
        fileMenu = Menu(menuBar)
        fileMenu.add_command(label="Exit", command=self.onExit)
        menuBar.add_cascade(label="File", menu=fileMenu)
  
  
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
         
       # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_OKsnapshot=tkinter.Button(window, text="OK", width=50, command=self.process)
        self.btn_OKsnapshot.pack(anchor=tkinter.CENTER, expand=True)
         # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
    
        self.window.mainloop()
    def show_entry_fields():
        print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
        e1.delete(0,END)
        e2.delete(0,END)
    def onExit(self):
        self.window.destroy()
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            im_arr.append(frame)
    def process(self):
        self.vid.__del__()
        sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
        pnet, rnet, onet = mtcnn_detect.create_mtcnn(sess, 'models')
        cropped_im = []
        save_im = []
        directory = "./dataImg/" + inputValue +"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
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
                
                    cropped_im.append(cropped)
        print(directory)
        for i in range(len(save_im)):
            cv2.imwrite(directory+str(i)+'.jpg',save_im[i])
        print('Extracted: %d' % len(cropped_im))
        calSVM(cropped_im,inputValue)
        self.window.destroy()
        
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def calSVM(cropped_im,person_label):
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
            #accuracy = np.mean(np.equal(best_class_indices, y_test))
            #print('Accuracy: %.3f' % accuracy)
            with open('svm_classifier.pkl', 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved svm classifier')

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
a = []
def process():
    print(a)

    return 0
def retrie_input():
    global inputValue
    inputValue = textBox.get("1.0","end-1c")
    #root.quit()
    if inputValue in class_names:
        person_label = class_names.index(inputValue)
        messagebox.showinfo("Info", "This person has already been in database")
    else:
        person_label = len(class_names) 
        file = open('class.txt','w')  
        class_names.append(inputValue)
        for name in class_names:
            file.write(name + os.linesep)
        file.close()   
        messagebox.showinfo("Info", "This operation will add this person to database")
    App(tkinter.Toplevel(), "Tkinter and OpenCV")
    
    
    print(len(im_arr))
    print(a)
    return inputValue
def process():
    sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
    pnet, rnet, onet = mtcnn_detect.create_mtcnn(sess, 'models')
    cropped_im = []
    save_im = []
    directory = "./dataImg/" + inputValue +"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
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
                #cropped = prewhiten(cropped)
                #cropped = flip(cropped, False)
                cropped_im.append(cropped)
    print(directory)
    for i in range(len(save_im)):
        cv2.imwrite(directory+str(i)+'.jpg',save_im[i])
    print('Extracted: %d' % len(cropped_im))
    calSVM(cropped_im,inputValue)

# Create a window and pass it to the Aplication object
if __name__ == '__main__':
   root = Tk()
   root.geometry("200x100")
   textBox = Text(root, height = 1, width = 10)
   textBox.pack()
   textBox1 = Text(root, height = 1, width = 10)
   textBox1.pack()
   buttonCommit = Button(root, height = 1, width = 5, text = "Comfirm", command = lambda: retrie_input())
   buttonCommit.pack()
  
   root.mainloop()
   
   #App(tkinter.Toplevel(), "Tkinter and OpenCV")
   
