#!/usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager

import matplotlib
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QSlider, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg \
    import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt

matplotlib.use('Qt5Agg')


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = plt.figure(figsize=(5, 5))
        self.axes = fig.add_subplot(111, projection='3d')
        self.axes.set_xlim([-1, 1])
        self.axes.set_ylim([-1, 1])
        self.axes.set_zlim([0, 1])
        super(PlotCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.canvas = PlotCanvas(self)
        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider1.setMinimum(-90)
        self.slider1.setMaximum(90)
        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setMinimum(-90)
        self.slider2.setMaximum(90)
        self.slider3 = QSlider(Qt.Horizontal, self)
        self.slider3.setMinimum(-90)
        self.slider3.setMaximum(90)
        widget = QWidget()
        self.layout = QVBoxLayout()
        widget.setLayout(self.layout)
        self.layout.addWidget(self.slider1)
        self.layout.addWidget(self.slider2)
        self.layout.addWidget(self.slider3)
        self.layout.addWidget(self.canvas)
        self.setCentralWidget(widget)

        self.n_frames = 50

        self.tm = UrdfTransformManager()
        filename = "./hexapod_leg.urdf"
        with open(filename, "r") as f:
            robot_urdf = f.read()
            self.tm.load_urdf(robot_urdf, mesh_path="./")

        self.tm.set_joint("joint2", 0)
        self.tm.set_joint("joint3", 0.35 * np.pi)
        self.tm.set_joint("joint4", 0.2 * np.pi)
        self.tm.plot_visuals("hexapod_leg", ax=self.canvas.axes,
                             ax_s=0.6, alpha=1)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        q1 = self.slider1.value()*math.pi/180
        q2 = self.slider2.value()*math.pi/180
        q3 = self.slider3.value()*math.pi/180

        self.canvas.axes.cla()

        self.tm.set_joint("joint2", q1)
        self.tm.set_joint("joint3", q2)
        self.tm.set_joint("joint4", q3)

        self.tm.plot_visuals("hexapod_leg", ax=self.canvas.axes,
                             ax_s=0.6, alpha=1)
        self.canvas.axes.set_xlim([-1, 1])
        self.canvas.axes.set_ylim([-1, 1])
        self.canvas.axes.set_zlim([0, 1])

        self.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
