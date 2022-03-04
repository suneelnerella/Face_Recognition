import face_recognition
import os
import cv2

known = 'known'
unknown = 'unknown'

print('Loading known faces...')
known_faces = []
known_names = []

def read_img(path):
    img = cv2.imread(path)
    (h,w) = img.shape[:2]
    width = 500
    ratio = width/float(w)
    height = int(h*ratio)
    return cv2.resize(img,(width,height))

for name in os.listdir(known):
    for filename in os.listdir(f'{known}/{name}'):
        image = read_img(f'{known}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print('Processing unknown faces...')
for filename in os.listdir(unknown):
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{unknown}/{filename}')
    locations = face_recognition.face_locations(image, model='cnn')
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, 0.7)
        match = None
        if True in results: 
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, 2)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)