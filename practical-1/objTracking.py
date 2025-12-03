import cv2

from Detector import detect
from KalmanFilter import KalmanFilter

font = 0  # cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (0, 0, 0)
fontThickness = 1

red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
cyan = (250, 250, 43)

radius = 15  # length of squares

dt = 0.1
u_x = 1
u_y = 1
std_acc = 1
x_std_meas = 0.1
y_std_meas = 0.1

filter = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

cap = cv2.VideoCapture("../data/randomball.avi")
trajectory = []
if not cap.isOpened():
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    centers = detect(frame)
    # assert(len(centers) == 0)
    center = centers[0]

    predict = filter.predict()
    predict_x = int(predict[0, 0])
    predict_y = int(predict[1, 0])

    correct = filter.update(center)
    correct_x = int(correct[0, 0])
    correct_y = int(correct[1, 0])

    trajectory.append((correct_x, correct_y))

    # Print trajectory
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], cyan, 2)

    # Print prediction
    cv2.rectangle(
        frame,
        (predict_x - radius, predict_y - radius),
        (predict_x + radius, predict_y + radius),
        blue,
        2,
    )

    # Print correction
    cv2.rectangle(
        frame,
        (correct_x - radius, correct_y - radius),
        (correct_x + radius, correct_y + radius),
        red,
        2,
    )

    cv2.putText(
        frame, "Predicted Position", (10, 30), font, fontScale, blue, fontThickness
    )
    cv2.putText(
        frame, "Estimated Position", (10, 45), font, fontScale, red, fontThickness
    )

    # Display the resulting frame
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
