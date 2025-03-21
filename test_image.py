import cv2
import os

video_path = r"data/Queens_bev1.mp4"
video = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    if frame_count == 20:
        cv2.imshow("frame", frame[2])
        cv2.waitKey(0)
    frame_count+=1
video.release()
cv2.destroyAllWindows()