import cv2
import face_recognition
import numpy as np

video_capture = cv2.VideoCapture(0)


image = face_recognition.load_image_file("Your_image_for_detection.jpg")
face_encoding = face_recognition.face_encodings(image)[0]



known_face_encodings = [
    face_encoding,
]

known_face_names = [
    "Here_your_name",
]


face_locations = []
process_this_frame_every_second_time = True
face_encodings = []
face_names = []

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    if process_this_frame_every_second_time:

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame_every_second_time = not process_this_frame_every_second_time



    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(frame, (left, int(bottom - 35*(bottom - top)/200)), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, (bottom - top) / 200, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    esc = 27
    if cv2.waitKey(1) == esc:
        break

video_capture.release()
cv2.destroyAllWindows()