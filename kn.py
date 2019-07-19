
# python kn.py --model recogniser.model --labelbin labeler.pickle --image test3.jpg
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from classes.nms import non_max_suppression_slow
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])
mser = cv2.MSER_create()

img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()
rects = []
regions = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
for i, contour in enumerate(hulls):
    x,y,w,h = cv2.boundingRect(contour);rects.append((x, y, x+w, y+h))
	
pick = non_max_suppression_slow(np.array(rects), 0.3)
text=""
lb = pickle.loads(open(args["labelbin"], "rb").read())
model = load_model(args["model"])
i=0
results = []
for (startX, startY, endX, endY) in pick:
			cv2.rectangle(vis, (startX, startY), (endX, endY), (0, 255, 0), 2)
			image = img[startY: endY,startX: endX].copy()
		
			#cv2.imwrite('{}.png'.format(i), img[startY:endY,startX:endX])
		
			#cv2.imshow("cropped", image)
			#cv2.waitKey(0)
			#cv2.imshow("fsk",image)
			#cv2.waitKey(0)
			image = cv2.resize(image, (96, 96))
			image = image.astype("float") / 255.0
			image = img_to_array(image)
			image = np.expand_dims(image, axis=0)
			proba = model.predict(image)[0]
			idx = np.argmax(proba)
			label = lb.classes_[idx]
			filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
			#print(label)
			if (label=="Sample001"):
				label="0"
			if (label=="Sample002"):
				label="1"
			if (label=="Sample003"):
				label="2"
			if (label=="Sample004"):
				label="3"
			if (label=="Sample005"):
				label="4"
			if (label=="Sample006"):
				label="5"
			if (label=="Sample007"):
				label="6"
			if (label=="Sample008"):
				label="7"
			if (label=="Sample009"):
				label="8"
			if (label=="Sample010"):
				label="9"
			if (label=="Sample011"):
				label="A"
			if (label=="Sample012"):
				label="B"
			if (label=="Sample013"):
				label="C"
			if (label=="Sample014"):
				label="D"
			if (label=="Sample015"):
				label="E"
			if (label=="Sample016"):
				label="F"
			if (label=="Sample017"):
				label="G"
			if (label=="Sample018"):
				label="H"
			if (label=="Sample019"):
				label="I"
			if (label=="Sample020"):
				label="J"
			if (label=="Sample021"):
				label="K"
			if (label=="Sample022"):
				label="L"
			if (label=="Sample023"):
				label="M"
			if (label=="Sample024"):
				label="N"
			if (label=="Sample025"):
				label="O"
			if (label=="Sample026"):
				label="P"
			if (label=="Sample027"):
				label="Q"
			if (label=="Sample028"):
				label="R"
			if (label=="Sample029"):
				label="S"
			if (label=="Sample030"):
				label="T"
			if (label=="Sample031"):
				label="U"
			if (label=="Sample032"):
				label="V"
			if (label=="Sample033"):
				label="W"
			if (label=="Sample034"):
				label="X"
			if (label=="Sample035"):
				label="Y"
			if (label=="Sample036"):
				label="Z"
			if (label=="Sample037"):
				label="a"
			if (label=="Sample038"):
				label="b"
			if (label=="Sample039"):
				label="c"
			if (label=="Sample040"):
				label="d"
			if (label=="Sample041"):
				label="e"
			if (label=="Sample042"):
				label="f"
			if (label=="Sample043"):
				label="g"
			if (label=="Sample044"):
				label="h"
			if (label=="Sample045"):
				label="i"
			if (label=="Sample046"):
				label="j"
			if (label=="Sample047"):
				label="k"
			if (label=="Sample048"):
				label="l"
			if (label=="Sample049"):
				label="m"
			if (label=="Sample050"):
				label="n"
			if (label=="Sample051"):
				label="o"
			if (label=="Sample052"):
				label="p"
			if (label=="Sample053"):
				label="q"
			if (label=="Sample054"):
				label="r"
			if (label=="Sample055"):
				label="s"
			if (label=="Sample056"):
				label="t"
			if (label=="Sample057"):
				label="u"
			if (label=="Sample058"):
				label="v"
			if (label=="Sample059"):
				label="w"
			if (label=="Sample060"):
				label="x"
			if (label=="Sample061"):
				label="y"
			if (label=="Sample062"):
				label="z"
			print (label)
			cv2.putText(vis, label, (startX, startY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
			results.append(((startX, startY, endX, endY), label))
			i=i+1
			text = text + label
results = sorted(results, key=lambda r:r[0][1])
 

"""
for ((startX, startY, endX, endY), label) in results:
	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(label))
"""	
			

print (text)
vis = imutils.resize(vis, width=400)
cv2.imshow("Output", vis)
cv2.waitKey(0)