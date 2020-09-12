import cv2

video = cv2.VideoCapture('pedestrain_5.mp4')
# pre trained car classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'pedestrian_tracking.xml'
# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    (read_successful, frame) = video.read()
    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 1400, 800)
    # display img
    cv2.imshow('output', frame)
    # wait for key press
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

video.release()
# convert to grayscale
# bnw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # create car classifier
# car_tracker = cv2.CascadeClassifier(classifier_file)
#
# # detect cars
# cars = car_tracker.detectMultiScale(bnw)

# for (x, y, w, h) in cars:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


print('Code Completed')
