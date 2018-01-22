import csv
import cv2
import argparse
import pandas
from pandas import DataFrame,read_csv


startX = []
startY = []
endX = []
endY = []

def clickEvent(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        startX.append(x)
        startY.append(y)
    elif event == cv2.EVENT_LBUTTONUP:
        endX.append(x)
        endY.append(y)

        cv2.rectangle(image, (startX[-1], startY[-1]), (endX[-1],endY[-1]), (0,255,0), 1)
        cv2.imshow("Image", image)

a = argparse.ArgumentParser()
a.add_argument("-i","--image",required=True,help="Image of the template scantron")
args = vars(a.parse_args())

image = cv2.imread(args["image"])
h,w = image.shape[:2]
h = int(h*0.78)
w = int(w*0.78)
image = cv2.resize(image,(w,h),interpolation=cv2.INTER_CUBIC)
clone = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image",clickEvent)

while True:
    cv2.imshow("Image",image)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('r'):
        image = clone.copy()
        startX = []
        startY = []
        endX = []
        endY = []

    if key == ord('c'):
        break

dims = [h,w]
dims.append(0)
dims.append(0)
pointsArray = [(dims[0],dims[1],dims[2],dims[3])]
for i in range(0,len(startX)):
    pointsArray.append((startX[i],startY[i],endX[i],endY[i]))

df = pandas.DataFrame(data = pointsArray, columns=["StartX","StartY","EndX","EndY"])
print df

df.to_csv("QuestionBoxBounds.csv",index=False,header=False)

cv2.destroyAllWindows()
