#This will be an object oriented version
#of the virtual3d game
import cv2
import numpy

print('Starting OO virtual3d')


class Tunnel:
	pass


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
		cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 255), 5)
		return (bx+bw/2),(by+bh/2)


#-------------------------------------
#main
#

ff = FaceFinder()
cap = cv2.VideoCapture(cv2.CAP_ANY)
if not cap.isOpened():
	print('couldnt open cam')
	exit()




while True:
  retval, frame = cap.read()
  if retval == False:
    print('camera error!')

  ff.find_face(frame)
  cv2.imshow('q to quit', frame)

  if cv2.waitKey(30) == ord('q'):
    break



pause = input('press enter to end')

cap.release()

cv2.destroyAllWindows()


print('virtual3d complete')
