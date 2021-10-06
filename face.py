import cv2
import face_recognition
import numpy as np

# Доступ к камере.
video_capture = cv2.VideoCapture(0)


image = face_recognition.load_image_file("Your_image_for_detection.jpg")
face_encoding = face_recognition.face_encodings(image)[0]


# Массив с энкодиннгами изображений, которые можно распознавать.
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

    # Кадр с камеры.
    ret, frame_bgr = video_capture.read()

    rgb_frame = frame_bgr[:, :, ::-1]

    # Берем каждый второй кадр для экономии.
    if process_this_frame_every_second_time:

        # Поиск всех лиц. (Для Классификатора).
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Распознаные лица.
        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Смотрим на дистанции (Насколько лицо с камеры совпадает с доступным лицом в списке known_face_encodings).
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    # Чтобы обрабатывать только каждый второй кадр.
    process_this_frame_every_second_time = not process_this_frame_every_second_time



    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(frame_bgr, (left, int(bottom - 35*(bottom - top)/200)), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_bgr, name, (left + 6, bottom - 6), font, (bottom - top) / 200, (255, 255, 255), 1)

    # Вывод преобразованного изображения
    cv2.imshow('Video', frame_bgr)

    esc = 27
    if cv2.waitKey(1) == esc:
        break

video_capture.release()
cv2.destroyAllWindows()