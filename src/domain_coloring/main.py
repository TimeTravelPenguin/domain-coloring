import sys
from collections.abc import Callable
from typing import Literal

import numpy as np
from matplotlib import colors
from numpy.typing import NDArray
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

LevelCurve = Literal["None", "Modulus", "Phase", "Both"]


def ease(x: float) -> float:
    # Easing function for smooth animation
    return 2 * x * x if x < 0.5 else 1 - np.pow(-2 * x + 2, 2) / 2


def approx_deriv(f: Callable, x: float, h: float = 1e-5) -> float:
    # Approximate derivative for animation speed
    return (f(x + h) - f(x - h)) / (2 * h)


def brightness_for_level_curves(levelcurve: LevelCurve, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    # Calculate brightness for level curves
    result = np.ones_like(magnitude)
    if levelcurve in ("Modulus", "Both"):
        log_magnitude = np.log(np.abs(magnitude))
        result *= 0.655 + 0.3 * (log_magnitude - np.floor(log_magnitude))

    if levelcurve in ("Phase", "Both"):
        result *= 0.655 + 0.3 * (20 * phase - np.floor(20 * phase))

    return result


def enhanced_phase_portrait(Z: NDArray, func: Callable, level_curves: LevelCurve) -> NDArray:
    """
    Generate an enhanced phase portrait of the complex function over grid Z.
    """
    # Compute the function values
    F = func(Z)

    # Compute magnitude and phase
    magnitude = np.abs(F)
    phase = np.pi + np.arctan2(F.imag, -F.real)

    # Map phase to [0, 1] for hue
    hue = phase / (2 * np.pi)

    # Calculate brightness with level curves
    # epsilon = 1e-10
    # magnitude += epsilon
    brightness = brightness_for_level_curves(level_curves, magnitude, hue)

    # Ensure brightness is in [0,1]
    brightness = np.clip(brightness, 0, 1)

    # Set saturation to maximum
    saturation = np.ones_like(hue)

    # Create HSV image
    HSV = np.zeros((*Z.shape, 3))
    HSV[..., 0] = hue
    HSV[..., 1] = saturation
    HSV[..., 2] = brightness

    # Convert HSV to RGB
    RGB = colors.hsv_to_rgb(HSV)

    return RGB


class EnhancedPhasePortraitApp(QWidget):
    def __init__(self):
        super().__init__()

        self.frames = {}

        # Set window properties
        self.setWindowTitle("Enhanced Phase Portrait Plotter")
        self.setMinimumSize(200, 200)  # Allow window to be resized smaller
        self.resize(800, 600)

        # Create layouts
        layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        t_slider_layout = QHBoxLayout()

        # Function input field
        self.function_input = QLineEdit(self)
        self.function_input.setPlaceholderText("Enter function of z, e.g., z**2")
        self.function_input.returnPressed.connect(self.plot_function)
        self.function_input.setText("z * (1 - t) + t * (z - 1)/(z ** 2 + z + 1)")
        self.function_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.function_input.setMinimumWidth(50)

        # Level curves combo box
        self.level_curves = QComboBox(self)
        self.level_curves.addItem("None", "None")
        self.level_curves.addItem("Modulus", "Modulus")
        self.level_curves.addItem("Phase", "Phase")
        self.level_curves.addItem("Both", "Both")
        self.level_curves.setCurrentIndex(0)
        self.level_curves.currentIndexChanged.connect(self.plot_function)
        self.level_curves.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.level_curves.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        # Slider for parameter t
        self.t_slider_max = 1000
        self.t_slider_dir = 1
        self.t_slider = QSlider(Qt.Horizontal, self)
        self.t_slider.setMinimum(0)
        self.t_slider.setMaximum(self.t_slider_max)
        self.t_slider.setValue(0)
        self.t_slider.setTickPosition(QSlider.TicksBelow)
        self.t_slider.setTickInterval(self.t_slider_max // 10)
        self.t_slider.valueChanged.connect(self.update_t_label)
        self.t_slider.valueChanged.connect(self.plot_function)
        self.t_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Label for t value
        self.t_slider_label = QLabel(f"t: {self.t_slider.value() / self.t_slider_max:.2f}", self)
        self.t_slider_label.setAlignment(Qt.AlignCenter)
        # self.t_slider_label.

        # Button to animate t
        self.animate_t_slider_button = QPushButton("Animate t", self)
        self.animate_t_slider_button.setCheckable(True)
        self.animate_t_slider_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        # Plot button
        self.plot_button = QPushButton("Plot", self)
        self.plot_button.clicked.connect(self.plot_function)
        self.plot_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        # Image label to display the plot
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(0, 0)  # Allow image label to shrink

        # Add widgets to layouts with stretch factors
        input_layout.addWidget(self.function_input, 1)
        input_layout.addWidget(self.level_curves)

        t_slider_layout.addWidget(self.t_slider_label)
        t_slider_layout.addWidget(self.t_slider, 1)
        t_slider_layout.addWidget(self.animate_t_slider_button)

        layout.addLayout(input_layout)
        layout.addLayout(t_slider_layout)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.image_label)

        # Set the layout for the main window
        self.setLayout(layout)

        # Timer to handle resize events
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.frames.clear)
        self.resize_timer.timeout.connect(self.plot_function)

        # Initialize animation variables
        self.animate_t_slider = False
        self.pause_t_slider = False

        # Connect animate button
        self.animate_t_slider_button.clicked.connect(self.on_animate_button_click)

        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_t)
        self.timer.start(10)

        # Timer for pause in animation
        self.pause_timer = QTimer(self)
        self.pause_timer.timeout.connect(lambda: setattr(self, "pause_t_slider", False))
        self.pause_timer.setSingleShot(True)

    def update_t_label(self):
        # Update the t value label
        t_value = self.t_slider.value()
        self.t_slider_label.setText(f"t: {t_value / self.t_slider_max:.2f}")

    def on_animate_button_click(self):
        if not self.animate_t_slider and self.t_slider.value() == 0:
            self.t_slider.setValue(1)
            self.t_slider_dir = 1
            self.pause_t_slider = False
        elif not self.animate_t_slider and self.t_slider.value() == self.t_slider_max:
            self.t_slider.setValue(self.t_slider_max - 1)
            self.t_slider_dir = -1
            self.pause_t_slider = False

        self.animate_t_slider = self.animate_t_slider_button.isChecked()

    def animate_t(self):
        if self.animate_t_slider and not self.pause_t_slider:
            if self.t_slider.value() >= self.t_slider_max:
                self.t_slider_dir = -1
                self.t_slider.setValue(self.t_slider_max)
                self.pause_t_slider = True
                self.pause_timer.start(1000)
            elif self.t_slider.value() <= 0:
                self.t_slider_dir = 1
                self.t_slider.setValue(0)
                self.pause_t_slider = True
                self.pause_timer.start(1000)

            smooth_step = max(
                1,
                approx_deriv(ease, (self.t_slider.value() / self.t_slider_max)) * np.sqrt(self.t_slider_max),
            )
            new_t = round(self.t_slider.value() + self.t_slider_dir * smooth_step)
            self.t_slider.setValue(new_t)

    def plot_function(self):
        # Get the function string from input
        func_str = self.function_input.text()
        if not func_str:
            return

        if (data := self.level_curves.currentData()) in self.frames and self.t_slider.value() in self.frames[data]:
            try:
                self.image_label.setPixmap(self.frames[self.t_slider.value()])
            except Exception as e:
                error_message = f"Error displaying frame:\n{e}"
                QMessageBox.critical(self, "Error", error_message)
            return

        # Define the complex function
        try:

            def func(z):
                return eval(
                    func_str, {"z": z, "t": self.t_slider.value() / self.t_slider_max, "np": np, "__builtins__": {}}
                )
        except Exception as e:
            self.animate_t_slider = False
            error_message = f"Error parsing function:\n{e}"
            QMessageBox.critical(self, "Error", error_message)
            return

        try:
            # Get the size of the image label's contentsRect
            size = self.image_label.contentsRect().size()
            width = size.width()
            height = size.height()

            # If size is zero, do nothing
            if width <= 0 or height <= 0:
                return

            # Limit the maximum resolution
            max_N = 1000
            N_width = int(min(width, max_N))
            N_height = int(min(height, max_N))

            # Compute aspect ratio
            aspect_ratio = width / height

            # Adjust function bounds based on aspect ratio
            y_min = -5
            y_max = 5
            y_range = y_max - y_min

            x_center = 0
            x_range = y_range * aspect_ratio
            x_min = x_center - x_range / 2
            x_max = x_center + x_range / 2

            # Create a grid of complex numbers
            re = np.linspace(x_min, x_max, N_width)
            im = np.linspace(y_min, y_max, N_height)
            RE, IM = np.meshgrid(re, im)
            Z = RE + 1j * IM

            # Compute the enhanced phase portrait
            RGB = enhanced_phase_portrait(Z, func, self.level_curves.currentData())

            # Flip the image vertically for correct orientation
            RGB = np.flipud(RGB)

            # Convert RGB array to QImage
            height, width, channel = RGB.shape
            bytes_per_line = 3 * width
            RGB_255 = (RGB * 255).astype("uint8")

            # Create QImage and scale to fit label
            qImg = QImage(RGB_255.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)

            # Display the image scaled to the label's size
            self.frames[self.t_slider.value()] = pixmap
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            error_message = f"Error plotting function:\n{e}"
            QMessageBox.critical(self, "Error", error_message)
            return

    def resizeEvent(self, event):
        # Start the timer to delay replotting after resizing
        self.resize_timer.start(100)  # Wait 100 ms after resizing stops
        super().resizeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    ex = EnhancedPhasePortraitApp()
    ex.plot_function()

    # Shortcut to quit the application with Esc key
    QShortcut(Qt.Key_Escape, ex, ex.close)

    ex.show()
    sys.exit(app.exec_())
