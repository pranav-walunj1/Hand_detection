import socket
import time
import cv2
import numpy as np
import glob
import mediapipe as mp

# import SocketServer
img_array = []
host, port = "127.0.0.1", 25001
data = "HIIIII, THIS IS CLIENT!!!"

# SOCK_STREAM means TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils


def handDetection():
    count = 1
    path = 'D:\Pycharm Projects\pythonClient\images\\'
    ext = '.jpg'
    file = path + str(count) + ext
    fp = open('HandCordinates', 'w')
    while count <= 500:
        for filename in glob.glob(file):
            print(filename)
            img = cv2.imread(filename)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_hands.Hands().process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    fp.write(hand_landmark)


        count = count + 1
        file = path + str(count) + ext

    fp.close()


def createVideo():
    count = 1
    path = 'D:\Pycharm Projects\pythonClient\images\\'
    ext = '.jpg'
    file = path + str(count) + ext
    while count <= 500:
        for filename in glob.glob(file):
            print(filename)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        count = count + 1
        file = path + str(count) + ext

    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


#
# try:
#     # Connect to the server and send the data
#     sock.connect((host, port))
#     sock.sendall(data.encode("utf-8"))
#     response = sock.recv(1024)
#     print(response)
#     count = 1
#     imgname = str(count) + '.jpg'
#     var = 'images'+ "\\" + imgname
#     print(var)
#
#     while count <= 500:
#
#         fp = open(var, 'wb')
#
#         #time.sleep(3)
#         string1 = sock.recv(68438)
#
#         print(string1)
#
#         #var = str(input('DO you want to receive more data ? [Y/N]'))
#         #if var == 'N':
#          #  break
#         fp.write(string1)
#         fp.close()
#         count = count + 1
#         imgname = str(count) + '.jpg'
#         var = 'images' + "\\" + imgname
#         print(var)
#
#     print('Data Successfully received!!! ')
#     createVideo()
#     print('VIDEO CREATED!!!')
#
# finally:
#     sock.close()


handDetection()
