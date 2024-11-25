import cv2
import numpy

print('Starting OO virtual3d')

class FaceFinder:
	"""Use haar cascade filter to detect largest face from a frame."""

	def __init__(self):
		print('Face Finder initialize')
		self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	def find_face(self,frame):
		"""Returns face center(x,y), draws Rect on frame"""
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, minNeighbors = 9)

		if faces is None:
			return None
		bx=by=bw=bh=0

		for (x,y,w,h) in faces:
			if w > bw:
				bx,by,bw,bh = x,y,w,h
		cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 5)
		return (bx+bw/2),(by+bh/2)




#-------------------------------------
#main
#

ff = FaceFinder()
print('virtual3d complete')
