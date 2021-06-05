import cv2
import numpy as np
from pose.detector import PoseDetector
from drawer import Drawer


if __name__ == '__main__':
    cap = cv2.VideoCapture(0) # либо можно передать путь до видеофайла
    success, frame = cap.read()
    if not success:
        print("Ошибка при чтении видео")
        exit(0)
    
    detector = PoseDetector("nn/FP16-INT8/human-pose-estimation-0001.xml", frame_shape=frame.shape)
    drawer = Drawer()

    while True:
        success, frame = cap.read()
        if not success:
            print("Видео закончилось")
            break

        poses = detector.detect(frame)
        frame = drawer.draw_poses(img=frame, poses=poses)
        text = "FPS: " + ("%.1f" % detector.get_fps())
        text += "\nPress Q for Exit"
        frame = drawer.draw_text(img=frame, text=text)
        cv2.imshow("Game", frame)
        key = cv2.waitKey(1)

        ESC_KEY = 27
        if key in {ord('q'), ord('Q'), ESC_KEY}:
            print("Похоже что вы устали. Ну ничего, заглядывайте в игру в следующий раз")
            break

