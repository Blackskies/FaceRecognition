import cv2

cascPath = "haarcascade_frontalface_default.xml"
eye_cascade = cv2.CascadeClassifier('eye.xml')
faceCascade = cv2.CascadeClassifier(cascPath)
faceProfiles = cv2.CascadeClassifier('haarcascade_smile.xml')
camera = cv2.VideoCapture(0)

while True:
	# Read the image
	ret, image = camera.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.3,
		minNeighbors=5,
		minSize=(10, 10)
	    #flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		roi_gray = gray[y: y+h, x:x+w]
		roi_color = image[y: y+h, x:x+w]
		eyes = faceProfiles.detectMultiScale(
			roi_gray,
			scaleFactor = 1.5,
			minNeighbors = 15,
			minSize = (30,30),
			maxSize = (100,100)
		)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)

	cv2.imshow("Faces found", image)

	m = cv2.waitKey(30) & 0xff
	if m == 27 :
		break

camera.release()
cv2.destroyAllWindows()