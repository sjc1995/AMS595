import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import image
import os
import cv2
import math
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import svm


def rgb2gray(rgbpic):
    return np.dot(rgbpic[...,:3],[0.299,0.587,0.114])

'''
#Gray process the picture, transfer the matrix into 2-D
#Formula: Gray_value=0.30*Red_value+0.59*Green_value+0.11*Blue_value
'''

def gray2bw(graypic):
    [m,n] = np.shape(graypic)
    bwpic = np.zeros([m,n])
    bwpic[graypic > np.mean(graypic)] = 0
    bwpic[graypic <= np.mean(graypic)] = 1
    return bwpic

'''
#Binaryzation, transfer the matrix into 2 value, with white background into 0,
#the character into 1, so we can distinguish the background and characters,
#In our case there are only two colors in the picture, and backgroung are
#usually white with value 255, and the characters' color are less than 255,
#so we transfer the value which is over the mean to 0 and less to 1.
'''

def line_cut(bw):
    [m,n] = np.shape(bw)
    line = bw.sum(axis=1)
    start = []
    end = []
    for i in range(m-1):
        if line[i] == 0 and line[i+1] > 0:
            start.append(i)
        elif line[i] > 0 and line[i+1] == 0:
            end.append(i)
    return [start,end]

def line_out(pic,start,end):
    result = []
    if len(list(start))!=len(list(end)):
        return 'wrong'
    else:
        for i in range(len(start)):
            if end[i]-start[i] > 3 :
                result.append(pic[start[i]:end[i],:])
        return result
'''
calculate the sum of each line, if it is 0, means the points in this row
are all white. We record the row of changing(from 0 to 1 means the start
of one line of word, and from 1 to 0 means the end of one line). And we
pick out the the matrix by line from each start to end. Then we get the
each word line.
'''

def column_cut(pic):
    [m,n] = np.shape(pic)
    column = pic.sum(axis=0)
    start = []
    end = []
    for i in range(n-1):
        if column[i] == 0 and column[i+1] > 0:
            start.append(i+1)
        elif column[i] > 0 and column[i+1] == 0:
            end.append(i)
    return [start,end]

def column_out(pic,start,end):
    result = []
    if len(list(start))!=len(list(end)):
        return 'wrong'
    else:
        for i in range(len(start)):
            if end[i] - start[i] > 1  :
                result.append(pic[:,start[i]:end[i]])
            if i < (len(start)-1):
                if start[i+1]-end[i] > 0.4 * np.shape(pic)[0]:
                    result.append(np.eye(10))
                if start[i+1]-end[i] > 0.8 * np.shape(pic)[0]:
                    result.append(np.eye(10))
        result.append(np.tril(np.ones([10,10])))
        return result

def column_out_t(pic,start,end):
    result = []
    if len(list(start))!=len(list(end)):
        return 'wrong'
    else:
        for i in range(len(start)):
            if end[i] - start[i] > 1  :
                result.append(pic[:,start[i]:end[i]])
            if i < (len(start)-1):
                if start[i+1]-end[i] > 0.25 * np.shape(pic)[0]:
                    result.append(np.eye(10))
        result.append(np.tril(np.ones([10,10])))
        return result
'''
calculate the sum of each column, if it is 0, means the points in this column
are all white. We record the row of changing(from 0 to 1 means the start
of a character, and from 1 to 0 means the end of a character). And we
pick out the the matrix by line from each start to end. Then we get the
each character.
if the distance between the end of last character with the start of next
character is over than 0.25*the height of the line, we record it as a space
with a 10*10 identity matrix(can be distinguished with other character) and
record a upper triangular matrix with all non-zeros elements is 1 as the sign
of changing to next line.
Due to space before and after the character of "l","i","I" are bigger than
others, so if the space is over 0.8*height of line, we record it as two spaces.
(Later the if the character is "l","i","I", we delete one space neighbor them,
if there are.
'''


def word_out(pic):
    line = pic.sum(axis=1)
    a = []
    for i in range(len(line)):
        if line[i] > 0:
            a.append(i)
    return pic[a[0]:a[-1],:]

'''
cut out the background in each picthre of character.
'''
def clearnoise(pic):
    [m,n] = np.shape(pic)
    noisei = []
    noisej = []
    for i in range(m-2):
        for j in range(n-2):
            if pic[i+1,j+1] == 1 and pic[i:(i+2),j:(j+2)].sum() <= 2:
                noisei.append(i+1)
                noisej.append(j+1)
    for i in range(len(noisei)):
        pic[noisei[i],noisej[i]] = 0
    return pic

'''
clear noise if there is less than one point in the neighbour of the point,
we though it is a noise point, we clear it.
'''

def recognize():
    file_name = input("Input the file name:")
    # input the file name
    picture = mpimg.imread(file_name)
    # read image
    gray_pic = rgb2gray(picture)
    bw_pic = gray2bw(gray_pic)
    bw_pic = clearnoise(bw_pic)
    # transfer into binaryzation and clear noise
    [line_start,line_end] = line_cut(bw_pic)
    line_pic = line_out(bw_pic,line_start,line_end)
    # cut into line
    result = []
    for i in range(len(line_pic)):
        [col_start,col_end] = column_cut(clearnoise(line_pic[i]))
        result = result + column_out(line_pic[i],col_start,col_end)
    # cut into characters
    final = []
    result1= []
    for i in range(len(result)):
        result1.append(word_out(result[i]))
        x = cv2.resize(result1[i],(10,10))
        final.append(np.reshape(x,100))
    # cut background and resize
    '''
    Cut the picture we want to recognize into characters and into a list
    '''
    picture_t = mpimg.imread("train.jpg")
    # read training image
    gray_pic_t = rgb2gray(picture_t)
    bw_pic_t = gray2bw(gray_pic_t)
    bw_pic_t = clearnoise(bw_pic_t)
    # transfere into binaryzation and clear noise
    [line_start_t,line_end_t] = line_cut(bw_pic_t)
    line_pic_t = line_out(bw_pic_t,line_start_t,line_end_t)
    #cut into line
    result_t = []
    for i in range(len(line_pic_t)):
        [col_start_t,col_end_t] = column_cut(line_pic_t[i])
        result_t = result_t + column_out_t(line_pic_t[i],col_start_t,col_end_t)
    #cut into character

    for i in range(len(result_t)):
        result_t[i] = word_out(result_t[i])
        x = cv2.resize(result_t[i],(10,10))
        result_t[i] = np.reshape(x,100)
    #cut background and resize
    '''
    Cut the training picture into characters and into a list
    '''
    target = np.array([1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9,0,10,0,11,0,12,0,13,0,14,0,15,0,16,0,17,0,18,0,19,0,20,0,21,57,22,0,23,0,24,0,25,0,26,0,53,0,54,0,55,0,56,57,
                       1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9,0,10,0,11,0,12,0,13,0,14,0,15,0,16,0,17,0,18,0,19,0,20,0,21,0,22,0,23,57,24,0,25,0,26,0,53,0,54,0,55,0,56,57,
                       1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9,0,10,0,11,0,12,0,13,0,14,0,15,0,16,0,17,0,18,0,19,0,20,0,21,0,22,0,23,0,24,0,25,57,26,0,53,0,54,0,55,0,56,57,
                       27,0,28,0,29,0,30,0,31,0,32,0,33,0,34,0,35,0,36,0,37,0,38,0,39,0,40,0,41,0,42,0,43,0,44,0,45,0,46,57,47,0,48,0,49,0,50,0,51,0,52,0,53,0,
                       54,0,55,0,56,57])
    clf = svm.SVC(gamma=100,C=0.8,kernel='linear',decision_function_shape='ovo')
    clf.fit(result_t,target)
    '''
    build the model
    '''
    pred = clf.predict(final)
    '''
    prediction
    '''
    pre = []
    for i in range(len(pred)):
        pre.append(pred[i])
        if pred[i] in [9,38,53,54]:
            [m,n] = np.shape(result1[i])
            if m/n < 4:
                pred[i] = 53
                pre[-1] = 53
            else:
                if i < 1 or pred[i-2] in [53,54,55,56]:
                    pred[i] = 9
                    pre[-1] = 9
                else:
                    pred[i] = 38
                    pre[-1] = 38
                    if pre[-2] == 0:
                       del pre[-2];
        # distingush l I .
        if pred[i] == 0 and pred[i-1]==38:
            del pre[-1]
        if pred[i] == 0 and pred[i-1]==35:
            del pre[-1]
        if pred[i] == 35 and pred[i-1] == 0:
            del pre[-2]
        if pred[i] == 0 and pred[i-1]==9:
            del pre[-1]
        if pred[i] == 9 and pred[i-1] == 0:
            del pre[-2]
        # deal with the space
        if pred[i] in [3,29]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=29
            else:
                pre[-1]=3
        if pred[i] in [15,41]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=41
            else:
                pre[-1]=15
        if pred[i] in [19,45]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=45
            else:
                pre[-1]=19
        if pred[i] in [21,47]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=47
            else:
                pre[-1]=21
        if pred[i] in [22,48]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=48
            else:
                pre[-1]=22
        if pred[i] in [23,49]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=49
            else:
                pre[-1]=23
        if pred[i] in [24,50]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=50
            else:
                pre[-1]=24
        if pred[i] in [26,52]:
            [m,n] = np.shape(result[i])
            v = result[i].sum(axis = 1)
            if sum(v[0:math.floor(m/6)]) == 0:
                pre[-1]=52
            else:
                pre[-1]=26
        # distinguish capital and lower
        if pred[i] == 38:
            w = word_out(result[i])
            [m,n] = np.shape(w)
            v = w.sum(axis = 1)
            for j in range(m):
                if v[j] == 0:
                    pre[-1] = 35
        # distinguish l and i
    return pre

a = [" ","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
     "a","b","c","d","e","f","g","h","i","j",'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','.',',','!','?','\n']
b = recognize()
result = ' '
for i in b:
    result += a[i]
print(result)
# plt.imshow(b)
# plt.show()
