# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import math


#After the program finds a box on the page (A single answer box for a question), it creates one of these objects
class QuestionBox(object):

	# Each question box has the (B&W) image of it, the contours found within the box, and its "ID"(Question Number)
	def __init__(self,image,ID):
		self.img = image
		self.conts = cv2.findContours(self.img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
		self.ID = ID

	# Approximates the bounding boxes of each bubble in a single row, given the bounding boxes of those bubbles found in the row
	def patchBubbleRow(self, bubble):
		numBubs = len(bubble)
		(b,a) = self.img.shape
		# Finds the average x, y, width, and height of the found bubbles to being approximation
		bounds = [cv2.boundingRect(c) for c in bubble]
		m = zip(*bounds)
		avg = [np.mean(m[x]) for x in range(0,4)]
		bubble = []

		# Starts from the bottom of the image, and stacks one on top of the other bubbles using average width and height
		for i in range(0,10):
			L = [[avg[0],b - b/10*i], [avg[0] + avg[2],b - b/10*i],[avg[0] + avg[2],b - b/10*(i+1)], [avg[0], b - b/10*(i+1)]]
			ctr = np.array(L).reshape((-1,1,2)).astype(np.int32)
			bubble.append(ctr)
		return bubble

	# Approximates the bounding boxes of each square which the student has handwritten a digit in, given the bounding boxes of the squares found
	def patchSquares(self, sqConts):
		# Number of squares found by the program already
		numSquares = len(sqConts)
		(b,a) = self.img.shape

		# Finds the average x, y, width, and height of the found squares to being approximation
		bounds = [cv2.boundingRect(c) for c in sqConts]
		m = zip(*bounds)
		avg = []
		if(numSquares == 0):
			avg = [5,340,40,40]
		else:
			avg = [np.mean(m[0]),np.mean(m[1]),np.mean(m[2]),np.mean(m[3])]
		# The error allowed before we can conclude that a certain square wasn't found
		# If there is not another square within 20% of the average width from the previous found square, we assume there was no square there
		epsilon = avg[3]/5

		# Starts from the bottom of the image, and stacks one on top of the other squares using average width and height
		i = 0
		sqConts = []
		while(i < 6):
			L = [[avg[0],b-i*avg[3]],[avg[0]+avg[2],b-i*avg[3]],[avg[0]+avg[2],b-(i-1)*avg[3]+5],[avg[0],b-(i-1)*avg[3]+5]]
			ctr = np.array(L).reshape((-1,1,2)).astype(np.int32)
			sqConts.append(ctr)
			i += 1
		return sqConts


	def findBubbles(self):
		bubbleConts = []
		sqConts = []
		for c in self.conts:
			(x,y,w,h) = cv2.boundingRect(c)
			ar = w/float(h)

			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			c = approx

			# From all the convex polygons found, it first determines if the bounding box of the shape is approximately square, and sufficiently large
			if(ar >= 0.85 and ar <= 1.15 and w >= 20 and h >= 20 and cv2.isContourConvex(c)):
				# If the polygon found is closer to the top of the image, it is likely a square
				if(x < 40):
					(l,m,n,o) = cv2.boundingRect(c)
					L = [[l+n,m],[l,m],[l,m+o],[l+n,m+o]]
					ctr = np.array(L).reshape((-1,1,2)).astype(np.int32)
					sqConts.append(ctr)
				else:
					bubbleConts.append(c)


		bubbleConts.sort(greater)
		estSqConts = self.patchSquares(sqConts)
		sqConts.sort(greater)
		estSqConts.sort(greater)

		# print("Number of squares found: "+str(len(sqConts)))
		# Determines confidence interval for squares
		iOus = []
		for a in sqConts:
			maxiOu = 0
			for b in estSqConts:
				iou = iOu(cv2.boundingRect(b),cv2.boundingRect(a))
				maxiOu = max(iou,maxiOu)
			iOus.append(maxiOu)
		#print("The square confidence is: %.2f" %max(iOus))
		if(len(sqConts) == 0):
			self.sqC = 0
		else:
			self.sqC = max(iOus)

		return (bubbleConts,sqConts)

	# Displays the question box
	def showImage(self):

		cv2.imshow("Image",self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# Returns the confidence levels determined using the iOu metric
	def determineConfidence(self):
		return [self.sqC, self.bubC]

	# Given the entire block of contours of bubbles, this determines which number was bubbled in.
	def findAnswerBubbled(self):

		bubbleConts = self.findBubbles()[0]
		# print("Number of bubbles found: "+str(len(bubbleConts)))

		s = 0.0
		bubbles = [[],[],[],[],[]]
		answers = [(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1)]
		xC = -1
		i = 0
		(b,a) = self.img.shape

		if(len(bubbleConts) == 0):
			self.bubC = 0
			return -1

		# Finds the average size of the bubbles found
		for c in bubbleConts:
			(x,y,w,h) = cv2.boundingRect(c)
			s += (w+h)/2
		avg = s/len(bubbleConts)

		# Separates the bubbles into their respective rows
		for c in bubbleConts:
			(x,y,w,h) = cv2.boundingRect(c)
			n = int((x)/avg) - 1
			if(n > 4):
				n = 4
			bubbles[n].append(c)


		for i in range(0,5):
			# If extraneous bubbles were found in a row, this was probably a bad row, so delete some bubbles
			# This is alright, as we are using the approximated bubbles anyways to grade
			while(len(bubbles[i]) > 10):
				bubbles[i].pop()

			for c in bubbles[i]:
				(m,n,o,p) = cv2.boundingRect(c)
				q = int((a-n)/avg) - 1
		# The approximated full row of 10 bubbles
		estBubble = [self.patchBubbleRow(bubbles[i]) for i in range(0,5)]

		# Determines confidence interval for bubbles
		iOuTotals = []
		for i in range(0,5):
			iOus = []
			for a in bubbles[i]:
				maxiOu = 0
				for b in estBubble[i]:
					iou = iOu(cv2.boundingRect(a),cv2.boundingRect(b))
					maxiOu = max(iou,maxiOu)
				iOus.append(maxiOu)
			iOuTotals.append(np.mean(iOus))
		#print("The bubble confidence is: %.2f" %max(iOuTotals))
		self.bubC = max(iOuTotals)

		for i in range(0,5):
			# This finds the most dark bubble in each row. In order to count as bubbled, the bubble must satisfy:
			# (1) It is sufficiently dark (threshold of 1200 pixels)
			# (2) If more than one bubble exceeds this threshold, an error will be announced
			bubbled = None
			for (j,c) in enumerate(estBubble[i]):
				mask = np.zeros(self.img.shape, dtype="uint8")
				cv2.drawContours(mask, [c], -1,255,-1)

				mask = cv2.bitwise_and(self.img, self.img, mask=mask)
				total = cv2.countNonZero(mask)

				if not(bubbled is None) and total > 1100:
					print("Error! There seems to be more than one bubble for question # "+str(self.ID))

				if (bubbled is None or total > bubbled[0]) and total > 1100:
					bubbled = (total, j)
			if not(bubbled is None):
				answers[i] = bubbled

		# answers[] will hold the darkness value, as well as the bubble that was found in each row.
		m = zip(*answers)

		#Ignores rows that weren't bubbled in (__235 is graded the same as 235__ is graded the same as _2_3_5)
		s = ""
		for x in m[1]:
			if x != -1:
				s = s + str(x)
		#print("The closest answer bubbled in for Question # "+str(self.ID)+" was: ")
		#print(s)
		return s






# Just a method to sort an array starting preference at the top left corner of the image
def greater(a, b):
	err = 20
	momA = cv2.moments(a)
	(xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

	momB = cv2.moments(b)
	(xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
	if(abs(xa-xb) > err):
		if(xa>xb):
			return 1
		else:
			return -1
	if(abs(xa-xb <= err)):
		if(ya > yb):
			return -1
		else:
			return 1


# Using intersection-over-union metric to determine how good our approximate bubbles and squares were
# http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iOu(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
	yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou




# The below code is heavily based off of http://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
# This code first finds the paper within the picture, then applies a transform to scale the paper to a rectangle (in case the picture was taken at an angle)
# It then finds all of the "question boxes" in the paper, and creates a list of the question boxes to apply grading procedures to

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())


# load the image, convert it to grayscale, blur it slightly, then find edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# find contours in the edge map, then initialize the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
	# sort the contours according to their size in descending order
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# loop over the sorted contours
	for c in cnts:
	# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we can assume we have found the paper
		if len(approx) == 4:
			docCnt = approx
			break

# apply a four point perspective transform to both the original image and grayscale image to obtain a top-down birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))




# apply Otsu's thresholding method to binarize the warped piece of paper
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and have an aspect ratio approximately equal to 1
	if w >= 20 and h >= 20:
		questionCnts.append(c)


questionImages = []

if len(questionCnts) > 0:
	# sort the contours according to their size in descending order
	questionCnts = sorted(questionCnts, key=cv2.contourArea, reverse=True)
	questionCnts.sort(greater)

	# loop over the sorted contours
	for c in questionCnts:
	# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we can assume we have found the paper
		if len(approx) == 4:
			questionImages.append(four_point_transform(thresh, approx.reshape(4, 2)))



# This part of the program is the class defined at the beginning of the program

questions = [(QuestionBox(x, questionImages.index(x)),False) for x in questionImages]
answers = []
confidences = []

for x in questions:
	answers.append(int(x[0].findAnswerBubbled()))
	if(int(x[0].findAnswerBubbled()) == -1):
		print("Error reading question #: "+str(x[0].ID))
	confidences.append(x[0].determineConfidence())
	#x[0].showImage()

print(answers)
n = zip(*confidences)
avSqC = np.mean(n[0])
avBubC = np.mean(n[1])

print("Square detection confidence: %.3f"%avSqC)
print("Bubble detection confidence: %.3f"%avBubC)
