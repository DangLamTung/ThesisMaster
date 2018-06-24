import tensorflow as tf
import mtcnn_detect
import cv2
import numpy as np
from scipy import misc
import time
from scipy.spatial import distance
from keras.models import load_model
import pickle
import facenet
# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
image_size = 160
  

sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = mtcnn_detect.create_mtcnn(sess, 'models')

with tf.Graph().as_default():
    with tf.Session() as sess:
        np.random.seed(seed= 1324 )
        facenet.load_model('./models')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        emb_array = np.zeros((1, embedding_size))
        (model, class_names) = pickle.load(open('svm_classifier.pkl', 'rb'))
        print(len(class_names))
        hn = cv2.VideoCapture(0)
        while True:
            _,frame = hn.read();
            bounding_boxes, _ = mtcnn_detect.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
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
                    cv2.rectangle(frame,(bb[i][0],bb[i][1]),(bb[i][2],bb[i][3]),(255,255,255)) #draw bounding box for the face
                    crop = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                    scaled[i] = cv2.resize(scaled[i], (image_size,image_size),interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,image_size,image_size,3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    for i in range(len(best_class_indices)):
                        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    cv2.putText(frame,class_names[best_class_indices[i]],(bb[i][0],bb[i][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)    
                    cv2.imshow(str(i),crop)
                cv2.imshow("Frame",frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
