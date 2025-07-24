import cv2
from MotionDetector import MotionDetector
import numpy as np
from skvideo.io import vread
from PySide6 import QtCore, QtWidgets, QtGui
import sys
import argparse

# Compatibility fix for skvideo
import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_

motionDetector = MotionDetector(
    tau=30, skip=1, max_objects=15, delta=50, alpha=5)


class QtDemo(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames
        self.current_frame = 3  # Start from the 4th frame
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(frames) - 1)

        self.next_button = QtWidgets.QPushButton("Next Frame")
        self.forward60 = QtWidgets.QPushButton("Forward 60")
        self.backward60 = QtWidgets.QPushButton("Backward 60")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.img_label)
        layout.addWidget(self.next_button)
        layout.addWidget(self.forward60)
        layout.addWidget(self.backward60)
        layout.addWidget(self.frame_slider)

        self.next_button.clicked.connect(self.on_click)
        self.forward60.clicked.connect(lambda: self.skip_frames(60))
        self.backward60.clicked.connect(lambda: self.skip_frames(-60))
        self.frame_slider.sliderMoved.connect(self.on_move)

        # Preload first 3 frames
        for i in range(3):
            motionDetector.add_frame(self.frames[i])
        self.frame_slider.setValue(self.current_frame)
        self.display_frame()

    def display_frame(self):
        frame = self.frames[self.current_frame].copy()

        # buffer frames to maintain context
        for i in range(max(0, self.current_frame - 2), self.current_frame):
            motionDetector.add_frame(self.frames[i])

        updated_frame = motionDetector.update(frame, self.current_frame)

        h, w, c = updated_frame.shape
        img = QtGui.QImage(updated_frame, w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

    @QtCore.Slot()
    def on_click(self):
        if self.current_frame >= len(self.frames) - 1:
            return
        self.current_frame += 1
        self.frame_slider.setValue(self.current_frame)
        self.display_frame()

    @QtCore.Slot()
    def on_move(self, pos):
        self.current_frame = pos
        self.display_frame()

    def skip_frames(self, delta):
        self.current_frame = max(
            0, min(len(self.frames) - 1, self.current_frame + delta))
        self.frame_slider.setValue(self.current_frame)
        self.display_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kalman filter tracking demo")
    parser.add_argument("video_path", type=str)
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--grey", type=str, default="False")
    args = parser.parse_args()

    if args.num_frames > 0:
        frames = vread(args.video_path, num_frames=args.num_frames,
                       as_grey=args.grey.lower() == "true")
    else:
        frames = vread(args.video_path, as_grey=args.grey.lower() == "true")

    print("[INFO] Loaded video with shape:", frames.shape)

    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    app = QtWidgets.QApplication([])
    widget = QtDemo(frames)
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())
