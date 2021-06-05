import logging
import sys
from pathlib import Path
import time

import cv2
import numpy as np
from openvino.inference_engine import IECore


from .open_pose import OpenPose
from .async_pipeline import AsyncPipeline
from pathlib import Path
from collections import deque


class PoseDetector:
    def __init__(self, model_path, frame_shape):
        aspect_ratio = frame_shape[1] / frame_shape[0]
        self.ie = IECore()
        self.model = OpenPose(self.ie, Path(model_path), target_size=None, aspect_ratio=aspect_ratio, prob_threshold=0.1)

        plugin_config = {"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}
        self.hpe_pipeline = AsyncPipeline(self.ie, self.model, plugin_config, device="CPU", max_num_requests=1)
        self.prev_poses = []
        self.last_dt = deque([], maxlen=10)

    def get_fps(self) -> float:
        if len(self.last_dt) == 0:
            return 0.0
        avg_dt = sum(self.last_dt) / float(len(self.last_dt))
        return 1.0 / avg_dt

    def detect(self, img) -> list:
        results = self.hpe_pipeline.get_result(0)
        if results:
            (poses, scores), frame_meta = results
            start_time = frame_meta['start_time']
            end_time = time.time()
            dt = end_time - start_time
            self.last_dt.append(dt)
            self.prev_poses = poses

        if self.hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = time.time()

            # Submit for inference
            self.hpe_pipeline.submit_data(img, 0, {'start_time': start_time})
            # next_frame_id += 1

        return self.prev_poses

        

