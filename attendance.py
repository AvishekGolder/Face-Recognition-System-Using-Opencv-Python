import cv2
import numpy as np 
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)
for cu_img in myList:
	current_image = cv2.imread(f'{path}/{cu_img}')
	images.append(current_image)
	personName.append(os.path.splitext(cu_img)[0])
print(personName)






def faceReading(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)

	return encodeList

readList = faceReading(images)
print("------------------- Data Read Success ---------------------")

def attendance(name):
	with open('data.csv', 'r+') as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			time_now = datetime.now()
			tStr = time_now.strftime('%H:%M:%S')
			dStr = time_now.strftime('%d/%m/%Y')
			

			if name == "RONALDO":
				ronroll = "1"
				f.writelines(f'\n{name},{tStr},{dStr},{ronroll}')

			if name == "WATSON":
				watroll = "2"
				f.writelines(f'\n{name},{tStr},{dStr},{watroll}')

			if name == "ZUCKERBURG":
				zuckroll = "3"
				f.writelines(f'\n{name},{tStr},{dStr},{zuckroll}')
			


cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
	faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

	facesCurrentFrame = face_recognition.face_locations(faces)
	encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

	for encodeFace, FaceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
		matches = face_recognition.compare_faces(readList, encodeFace)
		faceDis = face_recognition.face_distance(readList, encodeFace)

		matchIndex = np.argmin(faceDis)

		if matches[matchIndex]:
			name = personName[matchIndex].upper()
			

			y1,x2,y2,x1 = FaceLoc
			y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
			cv2.rectangle(frame, (x1,y2), (x2,y2), (0,128,128), 2)
			cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,128,128), cv2.FILLED)
			cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (192,192,192), 2)
			attendance(name)
	cv2.imshow("Face Recognition System", frame)
	if cv2.waitKey(10) == 32:
		break
cap.release()
cv2.destroyAllWindows()
