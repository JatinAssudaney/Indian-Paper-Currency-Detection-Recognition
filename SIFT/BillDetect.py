import numpy as np
import cv2
import common
import time
import threaded
import sys, getopt
import wave
import pyaudio
from common import anorm, getsize
import webbrowser
import os
import subprocess
from matplotlib import pyplot as plt
#--------------------------------------------------------------------------------------------------------
MIN_POINT = 30
chunk = 1024
FLANN_INDEX_KDTREE = 5

def init_feature():
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = MIN_POINT)
    matcher = cv2.BFMatcher(norm)
    return detector, matcher


    
def play_Sound(cash):
    if cash == '1':
        stream = p.open(format = p.get_format_from_width(sound20.getsampwidth()),  
                        channels = sound1.getnchannels(),  
                        rate = sound1.getframerate(),  
                        output = True)
        playnow = sound1
        
    
    elif cash == '2':
        stream = p.open(format = p.get_format_from_width(sound20.getsampwidth()),  
                        channels = sound2.getnchannels(),  
                        rate = sound2.getframerate(),  
                        output = True)
        playnow = sound2
    elif cash == '5':
        stream = p.open(format = p.get_format_from_width(sound50.getsampwidth()),  
                        channels = sound5.getnchannels(),  
                        rate = sound5.getframerate(),  
                        output = True)
        playnow = sound5
        
    elif cash == '10' or cash == '10,1' or cash == '10,2':
        stream = p.open(format = p.get_format_from_width(sound20.getsampwidth()),  
                        channels = sound10.getnchannels(),  
                        rate = sound10.getframerate(),  
                        output = True)
        playnow = sound10
        
    elif cash == '20':
        stream = p.open(format = p.get_format_from_width(sound20.getsampwidth()),  
                        channels = sound20.getnchannels(),  
                        rate = sound20.getframerate(),  
                        output = True)
        playnow = sound20
    elif cash == '50' or cash == '50,1':
        stream = p.open(format = p.get_format_from_width(sound50.getsampwidth()),  
                        channels = sound50.getnchannels(),  
                        rate = sound50.getframerate(),  
                        output = True)
        playnow = sound50
    elif cash == '100' or cash == '100,1' or cash == '100,2':
        stream = p.open(format = p.get_format_from_width(sound100.getsampwidth()),  
                        channels = sound100.getnchannels(),  
                        rate = sound100.getframerate(),  
                        output = True)
        playnow = sound100
        x = subprocess.Popen("notepad.exe information/file.txt")

#        webbrowser.open("file.txt")
    elif cash == '200':
        stream = p.open(format = p.get_format_from_width(sound500.getsampwidth()),  
                        channels = sound200.getnchannels(),  
                        rate = sound200.getframerate(),  
                        output = True)
        playnow = sound200
    
    elif cash == '500' or cash == '500,1' or cash == '500,2':
        stream = p.open(format = p.get_format_from_width(sound500.getsampwidth()),  
                        channels = sound500.getnchannels(),  
                        rate = sound500.getframerate(),  
                        output = True)
        playnow = sound500
        os.system("TASKKILL /F /IM notepad.exe")
        x = subprocess.Popen("notepad.exe information/file.txt")
        
    elif cash == '1000':
        stream = p.open(format = p.get_format_from_width(sound1000.getsampwidth()),  
                        channels = sound1000.getnchannels(),  
                        rate = sound1000.getframerate(),  
                        output = True)
        playnow = sound1000
        
    elif cash == '2000':
        stream = p.open(format = p.get_format_from_width(sound1000.getsampwidth()),  
                        channels = sound2000.getnchannels(),  
                        rate = sound2000.getframerate(),  
                        output = True)
        playnow = sound2000



    data = playnow.readframes(chunk)
    #play stream  
    while len(data) > 0:
        stream.write(data)
        data = playnow.readframes(chunk)


    playnow.rewind()
    #stop stream  
    stream.stop_stream()  
    stream.close()

    #close PyAudio  
    p.terminate()
    
def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), 2, color)

def filter_matches(kp1, kp2, matches, ratio = 0.75): #ratio = 0.75
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs


def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if H is not None :
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, showText, (corners[0][0],corners[0][1]), font, 1,(0,0,255),2)
    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
        
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs])
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = (0, 255, 0)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
    return vis

def match_and_draw(win, checksound, found , count):
    if(len(kp2) > 0):
	#matching feature
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= MIN_POINT:
            if not found:
                checksound = True
            found = True
            count = count + 1
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            matchesMask = status.ravel().tolist()
            vis = explore_match(win, img1, img2, kp_pairs, status, H)
            draw_keypoints(vis, kp1, color = (0, 255, 255))
            #	    print ("inliers/matched" + np.sum(status) + len(status))
            
        else:
            found = False
            checksound = False
            H, status = None, None
            count = 0
            #print '%d matches found, not enough for homography estimation' % len(p1)
            vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
            vis[:h1, :w1] = img1
            vis[:h2, w1:w1+w2] = img2
        
        cv2.imshow('find_obj', vis)
        if(count > 3):
            if checksound :
                checksound = False
                play_Sound(showText)
                count = 0
        
    return found, checksound ,count 
        
#--------------------------------------------------------------------------------------------------------
cv2.useOptimized()

cap = cv2.VideoCapture(0)
detector, matcher = init_feature()

checksound = True
found = False
searchIndex = 1
count = 0

#preLoad resource
img_source1 = cv2.imread( 'img_source/test1f.jpg' , 0 )
temp_kp1, temp_desc1 = detector.detectAndCompute(img_source1, None)

img_source2 = cv2.imread( 'img_source/test2f.jpg' , 0 )
temp_kp2, temp_desc2 = detector.detectAndCompute(img_source2, None)

img_source3 = cv2.imread( 'img_source/test3f.jpg' , 0 )
temp_kp3, temp_desc3 = detector.detectAndCompute(img_source3, None)

img_source4 = cv2.imread( 'img_source/test4.jpg' , 0 )
temp_kp4, temp_desc4 = detector.detectAndCompute(img_source4, None)

img_source5 = cv2.imread( 'img_source/test5.jpg' , 0 )
temp_kp5, temp_desc5 = detector.detectAndCompute(img_source5, None)

img_source6 = cv2.imread( 'img_source/test6.jpg' , 0 )
temp_kp6, temp_desc6 = detector.detectAndCompute(img_source6, None)

img_source7 = cv2.imread( 'img_source/test7.jpg' , 0 )
temp_kp7, temp_desc7 = detector.detectAndCompute(img_source7, None)

img_source8 = cv2.imread( 'img_source/test8.jpg' , 0 )
temp_kp8, temp_desc8 = detector.detectAndCompute(img_source8, None)

img_source9 = cv2.imread( 'img_source/test9.jpg' , 0 )
temp_kp9, temp_desc9 = detector.detectAndCompute(img_source9, None)

img_source10 = cv2.imread( 'img_source/test10.jpg' , 0 )
temp_kp10, temp_desc10 = detector.detectAndCompute(img_source10, None)

img_source11 = cv2.imread( 'img_source/test11.jpg' , 0 )
temp_kp11, temp_desc11 = detector.detectAndCompute(img_source11, None)

img_source12 = cv2.imread( 'img_source/test12.jpg' , 0 )
temp_kp12, temp_desc12 = detector.detectAndCompute(img_source12, None)

img_source13 = cv2.imread( 'img_source/test13.jpg' , 0 )
temp_kp13, temp_desc13 = detector.detectAndCompute(img_source13, None)

img_source14 = cv2.imread( 'img_source/test14.jpg' , 0 )
temp_kp14, temp_desc14 = detector.detectAndCompute(img_source14, None)

img_source15 = cv2.imread( 'img_source/test15.jpg' , 0 )
temp_kp15, temp_desc15 = detector.detectAndCompute(img_source15, None)

img_source16 = cv2.imread( 'img_source/test16.jpg' , 0 )
temp_kp16, temp_desc16 = detector.detectAndCompute(img_source16, None)

img_source17 = cv2.imread( 'img_source/test17.jpg' , 0 )
temp_kp17, temp_desc17 = detector.detectAndCompute(img_source17, None)

img_source18 = cv2.imread( 'img_source/test18.jpg' , 0 )
temp_kp18, temp_desc18 = detector.detectAndCompute(img_source18, None)

#preLoad sound

sound1 = wave.open(r"sound/1.wav","rb")
sound2 = wave.open(r"sound/2.wav","rb")
sound5 = wave.open(r"sound/5.wav","rb")
sound10 = wave.open(r"sound/10.wav","rb")
sound20 = wave.open(r"sound/20.wav","rb")
sound50 = wave.open(r"sound/50.wav","rb")
sound100 = wave.open(r"sound/100.wav","rb")
sound100 = wave.open(r"sound/100.wav","rb")
sound100 = wave.open(r"sound/100.wav","rb")
sound200 = wave.open(r"sound/200.wav","rb")
sound500 = wave.open(r"sound/500.wav","rb")
sound1000 = wave.open(r"sound/1000.wav","rb")
sound2000 = wave.open(r"sound/2000.wav","rb")

while(True):
    t1 = cv2.getTickCount()
    p = pyaudio.PyAudio() 
    #switch template
    if not found:
        if searchIndex <= 18:
            if searchIndex == 1:
                img1 = img_source1
                kp1 = temp_kp1
                desc1 = temp_desc1
                showText = '200'
            elif searchIndex == 2:
                img1 = img_source2
                kp1 = temp_kp2
                desc1 = temp_desc2
                showText = '50,1'
            elif searchIndex == 3:
                img1 = img_source3
                kp1 = temp_kp3
                desc1 = temp_desc3
                showText = '100,1'
            elif searchIndex == 4:
                img1 = img_source4
                kp1 = temp_kp4
                desc1 = temp_desc4
                showText = '10,1'
            elif searchIndex == 5:
                img1 = img_source5
                kp1 = temp_kp5
                desc1 = temp_desc5
                showText = '2'
            elif searchIndex == 6:
                img1 = img_source6
                kp1 = temp_kp6
                desc1 = temp_desc6
                showText = '500,1'
            elif searchIndex == 7:
                img1 = img_source7
                kp1 = temp_kp7
                desc1 = temp_desc7
                showText = '10'
            elif searchIndex == 8:
                img1 = img_source8
                kp1 = temp_kp8
                desc1 = temp_desc8
                showText = '20'
            elif searchIndex == 9:
                img1 = img_source9
                kp1 = temp_kp9
                desc1 = temp_desc9
                showText = '50'
            elif searchIndex == 10:
                img1 = img_source10
                kp1 = temp_kp10
                desc1 = temp_desc10
                showText = '5'
            elif searchIndex == 12:
                img1 = img_source12
                kp1 = temp_kp12
                desc1 = temp_desc12
                showText = '2000'
            elif searchIndex == 13:
                img1 = img_source13
                kp1 = temp_kp13
                desc1 = temp_desc13
                showText = '1000'
            elif searchIndex == 11:
                img1 = img_source11
                kp1 = temp_kp11
                desc1 = temp_desc11
                showText = '100'
            elif searchIndex == 14:
                img1 = img_source14
                kp1 = temp_kp14
                desc1 = temp_desc14
                showText = '500'
            elif searchIndex == 15:
                img1 = img_source15
                kp1 = temp_kp15
                desc1 = temp_desc15
                showText = '1'
            elif searchIndex == 16:
                img1 = img_source16
                kp1 = temp_kp16
                desc1 = temp_desc16
                showText = '10,2'
            elif searchIndex == 17:
                img1 = img_source17
                kp1 = temp_kp17
                desc1 = temp_desc17
                showText = '100,2'
            elif searchIndex == 18:
                img1 = img_source18
                kp1 = temp_kp18
                desc1 = temp_desc18
                showText = '500,2'
                
            searchIndex = searchIndex+1
        else:
            searchIndex = 1
            img1 = img_source1
            
                    
    #Capture frame-by-frame
    ret, frame = cap.read()
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    #calculate features
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print ('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    found, checksound, count = match_and_draw('find_obj',checksound,found,count)

    t2 = cv2.getTickCount()
    #calculate fps
    time = (t2 - t1)/ cv2.getTickFrequency()
    print ('FPS = ',1/time)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
# When everything done, release the capture
 
#close PyAudio  
p.terminate()
 
cap.release()
cv2.destroyAllWindows()
