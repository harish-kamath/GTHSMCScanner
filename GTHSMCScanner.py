from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import math
import pandas

MBE = 18
isScanned = False

def showImage(img,imgName="Image",resizeFactor=1):
    res = cv2.resize(img,None,fx=resizeFactor, fy=resizeFactor, interpolation = cv2.INTER_AREA)
    cv2.imshow(imgName,res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Paper(object):

    def __init__(self,imgPath):
        self.image = cv2.imread(imgPath,0)
        self.gray = self.image #cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.edged = cv2.Canny(blurred, 75, 200)

    def transformPaper(self):
        cnts = cv2.findContours(self.edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        docCnt = None
        foundDoc = False

        w,h = self.image.shape[:2]

        if len(cnts) > 0:
            cnts = sorted(cnts,key=cv2.contourArea,reverse = True)
            for c in cnts:
                peri = cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c,0.02*peri,True)

                if len(approx) == 4 and cv2.contourArea(c) > 0.4*(w*h):
                    docCnt = approx
                    foundDoc = True
                    break

        if not(foundDoc):
            raise Exception("Outline for Document not Found!")

        paper = four_point_transform(self.image, docCnt.reshape(4, 2))
        self.transformed = four_point_transform(self.gray, docCnt.reshape(4, 2))

        h,w = self.transformed.shape[:2]
        if w > h:
            self.transformed = imutils.rotate_bound(self.transformed,90)
        #showImage(self.transformed,"Transformed",0.3)
        #showImage(self.transformed)

        return self.transformed


    def applyStruct(self, csvPath):
        transImage = []
        if(not isScanned):
            transImage = self.transformPaper()
        else:
            transImage = self.image
        #showImage(transImage)
        df = pandas.read_csv(csvPath, header=None)
        newPaper = cv2.resize(transImage, (df[1][0], df[0][0]), interpolation=cv2.INTER_CUBIC)
        self.qBoxes = []
        for i in range(1,len(df[0])):
            startX = df[0][i]
            startY = df[1][i]
            endX = df[2][i]
            endY = df[3][i]
            im = newPaper[startY - MBE:endY + MBE, startX - MBE:endX + MBE]
            #h,w = im.shape[:2]

            #im = cv2.resize(im,(w*2,h*2),interpolation=cv2.INTER_CUBIC)
            m = QuestionBox(im,i-1)
            self.qBoxes.append(m)
            answer = m.gradeBox()
            print(answer)



class QuestionBox(object):
    def __init__(self, img,ID):
        self.image = img
        self.thresh = cv2.threshold(self.image, 0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        self.ID = ID

    def gradeBox(self):
        bubbleRows = self.findBubbles()
        answerList  = []
        for b in bubbleRows:
            nums = []
            hasBubbled = False
            i = 0
            for c in b:
                total = cv2.countNonZero(c)
                if total > 220:
                    if hasBubbled:
                        print("Multiple Bubbles Detected! Number: "+str(self.ID))
                    else:
                        nums.append(i)
                        hasBubbled = True
                i += 1
            if len(nums) > 0:
                answerList.append(nums[0])
        s = ""
        for i in answerList:
            s += str(i)
        return int(s)

    def findBubbles(self):
        cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        circles = []
        boxCnt = cnts[0]
        for c in cnts:
            (x,y,w,h) = cv2.boundingRect(c)
            ar = w/float(h)
            if(cv2.contourArea(c) > cv2.contourArea(boxCnt)):
                boxCnt = c
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            c = approx
            if w >= 12 and h >= 12 and ar >= 0.85 and ar <= 1.15 and cv2.isContourConvex(c) and len(approx) > 7:
                circles.append(c)
                #cv2.drawContours(self.image,[c],0,(255,255,0),-1)
                #showImage(self.image)

        cv2.drawContours(self.image, circles, -1, (255,255,0), -1)
        #showImage(self.image, "Box", 1)

        bubbleRows = self.separateRows(circles)
        estimatedBubbleRows = self.estimateRows(bubbleRows,cv2.boundingRect(boxCnt))
        #showImage(estimatedBubbleRows[0][3], "Box", 1)

        return estimatedBubbleRows

    def separateRows(self,circ):
        bounds = [cv2.boundingRect(c) for c in circ]
        w = np.mean([b[2] for b in bounds])
        h = np.mean([b[3] for b in bounds])

        bounds.sort(key=lambda bounds:bounds[1])
        groups = [[]]
        i = 0
        currY = bounds[0][1]
        for b in bounds:
            if abs(b[1]-currY <= h/4):
                groups[i].append(b)
            else:
                groups.append([])
                i += 1
                groups[i].append(b)
                currY = b[1]

        """
        for g in groups:
            if len(g) < 4:
                groups.remove(g)
        """
        #print(str(len(groups))+" rows found!")
        """
        rows = [[]]
        for g in groups:
            for c in g:
                rows[-1].append(self.image[c[1]:c[1]+c[3], c[0]:c[0]+c[2]])
            rows.append([])
        return rows
        """
        return groups

    def estimateRows(self,bR,rect):
        h,w = self.image.shape
        initX = rect[0]
        estB = []
        for r in bR:
            w = np.mean([x[2] for x in r])
            h = np.mean([y[3] for y in r])
            y = np.mean([z[1] for z in r])

            m = []
            for i in range(0,10):
                m.append((initX+w*i,y,w,h))
            estB.append(m)

        est = []
        for a in estB:
            m = []
            for b in a:
                (e,f,g,h) = b
                e = int(e)
                f = int(f)
                g = int(g)
                h = int(h)
                m.append(self.thresh[f:f+h,e:e+g])
            est.append(m)


        return est



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Path to the input image")
args = vars(ap.parse_args())
m = Paper(args["image"])
m.applyStruct("QuestionBoxBounds.csv")
