import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from matplotlib import colors
from numpy.typing import NDArray
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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

regex_t = re.compile(r"\bt\b")


def ease(x: float) -> float:
    # Easing function for smooth animation
    return 2 * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 2 / 2


def approx_deriv(f: Callable, x: float, h: float = 1e-5) -> float:
    # Approximate derivative for animation speed
    return (f(x + h) - f(x - h)) / (2 * h)


def brightness_for_level_curves(levelcurve: LevelCurve, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    # Calculate brightness for level curves
    result = np.ones_like(magnitude)
    if levelcurve in ("Modulus", "Both"):
        epsilon = 1e-10  # Prevent log(0)
        magnitude += epsilon
        log_magnitude = np.log(np.abs(magnitude))
        result *= 0.655 + 0.39 * (log_magnitude - np.floor(log_magnitude))

    if levelcurve in ("Phase", "Both"):
        result *= 0.655 + 0.39 * (20 * phase - np.floor(20 * phase))

    return result


def enhanced_phase_portrait(Z: NDArray, func: Callable, level_curves: LevelCurve) -> NDArray:
    """
    Generate an enhanced phase portrait of the complex function over grid Z.
    """
    # Compute the function values
    F = func(Z)

    # Compute magnitude and phase
    magnitude = np.abs(F)
    phase = np.pi + np.angle(F)

    # Map phase to [0, 1] for hue
    hue = phase / (2 * np.pi)

    # Calculate brightness with level curves
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
    frame_generated = pyqtSignal(int, QPixmap)  # Signal to communicate with the main thread

    def __init__(self):
        super().__init__()

        # Thread pool for generating frames
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.frames: dict[int, QPixmap] = {}

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
        self.function_input.returnPressed.connect(self.generate_frames)
        self.function_input.setText("(z - 1)/(z ** 2 + z + 1)")
        self.function_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.function_input.setMinimumWidth(50)

        # Level curves combo box
        self.level_curves = QComboBox(self)
        self.level_curves.addItem("None", "None")
        self.level_curves.addItem("Modulus", "Modulus")
        self.level_curves.addItem("Phase", "Phase")
        self.level_curves.addItem("Both", "Both")
        self.level_curves.setCurrentIndex(0)
        self.level_curves.currentIndexChanged.connect(self.generate_frames)
        self.level_curves.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.level_curves.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        # Slider for parameter t
        self.t_slider_max = 100
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
        self.t_slider_label.setFixedWidth(50)

        # Button to animate t
        self.animate_t_slider_button = QPushButton("Animate t", self)
        self.animate_t_slider_button.setCheckable(True)
        self.animate_t_slider_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.animate_t_slider_button.clicked.connect(self.on_animate_button_click)

        # Plot button
        self.plot_button = QPushButton("Plot", self)
        self.plot_button.clicked.connect(self.generate_frames)
        self.plot_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        # Image label to display the plot
        self.image_label = self.new_image_label()
        self.image_label_layout = QVBoxLayout()
        self.image_label_layout.addWidget(self.image_label)

        # Add widgets to layouts with stretch factors
        input_layout.addWidget(self.function_input, 1)
        input_layout.addWidget(self.level_curves)

        t_slider_layout.addWidget(self.t_slider_label)
        t_slider_layout.addWidget(self.t_slider, 1)
        t_slider_layout.addWidget(self.animate_t_slider_button)

        layout.addLayout(input_layout)
        layout.addLayout(t_slider_layout)
        layout.addWidget(self.plot_button)
        layout.addLayout(self.image_label_layout, 1)

        # Set the layout for the main window
        self.setLayout(layout)

        # Timer to handle resize events
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.generate_frames)

        # Initialize animation variables
        self.animate_t_slider = False
        self.pause_t_slider = False
        self.can_pause_t_slider = False

        # Connect frame_generated signal
        self.frame_generated.connect(self.on_frame_generated)

        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_t)
        self.timer.start(30)

        # Timer for pause in animation
        self.pause_timer = QTimer(self)
        self.pause_timer.timeout.connect(self.reset_pause_timer)
        self.pause_timer.setSingleShot(True)

    def new_image_label(self):
        image_label = QLabel(self)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setScaledContents(True)
        image_label.setMinimumSize(0, 0)
        return image_label

    def __del__(self) -> None:
        self.thread_pool.shutdown(wait=False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.on_animate_button_click()

    def reset_pause_timer(self):
        self.pause_t_slider = False

    def update_t_label(self):
        # Update the t value label
        t_value = self.t_slider.value()
        self.t_slider_label.setText(f"t: {t_value / self.t_slider_max:.2f}")

    def on_animate_button_click(self):
        if not self.animate_t_slider and self.t_slider.value() == 0:
            self.t_slider.setValue(1)
            self.t_slider_dir = 1
            self.pause_t_slider = False
            self.animate_t_slider = True
        elif not self.animate_t_slider and self.t_slider.value() == self.t_slider_max:
            self.t_slider.setValue(self.t_slider_max - 1)
            self.t_slider_dir = -1
            self.pause_t_slider = False
            self.animate_t_slider = True

        self.animate_t_slider = self.animate_t_slider_button.isChecked()

    def animate_t(self):
        if self.animate_t_slider and not self.pause_t_slider:
            if self.can_pause_t_slider and self.t_slider.value() >= self.t_slider_max:
                self.t_slider_dir = -1
                self.pause_t_slider = True
                self.can_pause_t_slider = False
                self.pause_timer.start(1000)
                self.t_slider.setValue(self.t_slider_max)
                return
            if self.can_pause_t_slider and self.t_slider.value() <= 0:
                self.t_slider_dir = 1
                self.pause_t_slider = True
                self.can_pause_t_slider = False
                self.pause_timer.start(1000)
                self.t_slider.setValue(0)
                return

            self.can_pause_t_slider = True

            smooth_step = max(
                1,
                approx_deriv(ease, (self.t_slider.value() / self.t_slider_max)) * np.sqrt(self.t_slider_max),
            )
            new_t = round(self.t_slider.value() + self.t_slider_dir * smooth_step)
            self.t_slider.setValue(new_t)

    def generate_frames(self):
        func_str = self.function_input.text()
        if not func_str:
            return

        # Clear existing frames
        self.frames.clear()
        # self.image_label.clear()

        # Define the complex function
        try:

            def func(z: np.ndarray, t: float) -> np.ndarray:
                func_str_local = func_str
                if func_str_local and regex_t.search(func_str_local) is None:
                    func_str_local = f"z * (1 - t) + t * ({func_str_local})"
                return eval(
                    func_str_local,
                    {"z": z, "t": t, "i": 1j, "x": z.real + 0j, "y": z.imag + 0j, "np": np, "__builtins__": {}},
                )
        except Exception as e:
            self.animate_t_slider = False
            error_message = f"Error parsing function:\n{e}"
            QMessageBox.critical(self, "Error", error_message)
            return

        # Get the size of the image label's contentsRect
        size = self.image_label.contentsRect().size()
        width = size.width()
        height = size.height()

        # If size is zero, do nothing
        if width <= 0 or height <= 0:
            return

        # Limit the maximum resolution
        # max_N = 500  # Adjusted for performance
        # N_width = int(min(width, max_N))
        # N_height = int(min(height, max_N))

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
        re = np.linspace(x_min, x_max, width)
        im = np.linspace(y_min, y_max, height)
        RE, IM = np.meshgrid(re, im)
        Z = RE + 1j * IM

        # Get current level curves setting
        level_curves = self.level_curves.currentData()

        # Submit tasks to thread pool
        for t_value in range(self.t_slider.minimum(), self.t_slider.maximum() + 1):
            self.thread_pool.submit(self.generate_frame, func, Z, level_curves, t_value)

    def generate_frame(self, func, Z, level_curves, t_value):
        t = t_value / self.t_slider_max

        # Define a function that only takes z
        def func_t(z):
            return func(z, t)

        try:
            RGB = enhanced_phase_portrait(Z, func_t, level_curves)

            # Flip the image vertically for correct orientation
            RGB = np.flipud(RGB)

            # Convert RGB array to QImage
            height, width, channel = RGB.shape
            bytes_per_line = 3 * width
            RGB_255 = (RGB * 255).astype("uint8")

            # Create QImage and scale to fit label
            qImg = QImage(RGB_255.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)

            # Emit signal to notify main thread
            self.frame_generated.emit(t_value, pixmap)
        except Exception as e:
            # Handle exceptions
            print(f"Error generating frame for t={t_value}: {e}")

    def on_frame_generated(self, t_value, pixmap):
        self.frames[t_value] = pixmap

        # If this frame corresponds to the current t_slider value, display it
        if t_value == self.t_slider.value():
            self.image_label.setPixmap(pixmap)

    def plot_function(self):
        t_value = self.t_slider.value()
        if t_value in self.frames:
            pixmap = self.frames[t_value]
            self.image_label.setPixmap(pixmap)
        else:
            # Optionally, clear the image or display a placeholder
            self.image_label.clear()

    def resizeEvent(self, event):
        # Start the timer to delay replotting after resizing
        self.resize_timer.start(100)  # Wait 100 ms after resizing stops
        super().resizeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    ex = EnhancedPhasePortraitApp()
    ex.generate_frames()

    # Shortcut to quit the application with Esc key
    QShortcut(Qt.Key_Escape, ex, ex.close)

    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
