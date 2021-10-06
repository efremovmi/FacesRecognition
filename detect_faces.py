from imutils.video import VideoStream
import cv2
import dlib

videoStream = VideoStream(src=0).start()

# Подключение детектора, настроенного на поиск лиц.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    frame = videoStream.read()

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(grayFrame)

    for face in faces:

        cv2.putText(frame, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Получение координат вершин прямоугольника и его построение на изображении
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Получение координат контрольных точек и их построение на изображении
        landmarks = predictor(grayFrame, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    cv2.putText(frame, "Press ESC to close frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Вывод преобразованного изображения
    cv2.imshow("VideoCap", frame)

    esc = 27
    if cv2.waitKey(1) == esc:
        break