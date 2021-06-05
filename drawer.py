import cv2
import numpy as np


class Drawer:
    def __init__(self, point_threshold = 0.1):
        self.default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
            (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

        self.colors = (
                (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
                (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
                (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
                (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
                (0, 170, 255))

        self.point_score_threshold = point_threshold

    
    def draw_poses(self, img, poses, stick_width = 4) -> np.array:
        if len(poses) == 0:
            return img

        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(int).tolist()
            points_scores = pose[:, 2]
            # Draw joints.
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > self.point_score_threshold:
                    cv2.circle(img, tuple(p), 1, self.colors[i], 2)
            # Draw limbs.
            for i, j in self.default_skeleton:
                if points_scores[i] > self.point_score_threshold and points_scores[j] > self.point_score_threshold:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=self.colors[j], thickness=stick_width)
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img

    def draw_text(self, img, text) -> np.array:
        origin = (0, 20)
        for i, line in enumerate(text.split('\n')):
            y = 24 + i*26
            cv2.putText(img, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.putText(img, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 80, 0), 2)
        
        return img