"""
Efficient Data Stream Anomaly Detection
Author: Rushil Deshwal

A real-time anomaly detection system that uses robust statistical methods to identify
anomalies in continuous data streams. The implementation focuses on efficiency and 
minimal external dependencies.

Dependencies:
- numpy
- matplotlib

Key Features:
- Adaptive statistical anomaly detection
- Real-time data stream processing
- Efficient sliding window implementation
- Dynamic threshold adjustment
- Visual monitoring capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Tuple, Deque
import time
from dataclasses import dataclass

matplotlib.use("Agg")


@dataclass
class DataPoint:
    """
    Represents a single data point in the stream.
    Attributes:
        value (float): The numerical value of the data point.
        timestamp (float): The time at which the data point was generated.
        is_anomaly (bool): Flag indicating if the point is classified as an anomaly.
        anomaly_score (float): The score indicating the anomaly severity.
    """

    value: float
    timestamp: float
    is_anomaly: bool = False
    anomaly_score: float = 0.0


class AnomalyDetector:
    """
    Adaptive Anomaly Detector with a sliding window of data points.
    Combines robust statistics and Exponentially Weighted Moving Average (EMA)
    for anomaly scoring and detection.
    """

    def __init__(self, window_size: int = 100, alpha: float = 0.1):
        """
        Initializes the AnomalyDetector with a specified window size and alpha for EMA.
        Args:
            window_size (int): Number of recent values to retain in the sliding window.
            alpha (float): Smoothing factor for EMA (between 0 and 1).
        """
        self.window_size = window_size
        self.alpha = alpha
        self.values = deque(maxlen=window_size)  # Store recent data points
        self.sorted_window = []  # Keep sorted values for robust statistics
        self.ema = None  # Exponentially Weighted Moving Average
        self.emstd = None  # Moving Standard Deviation for EMA

    def _update_sorted_window(self):
        """
        Maintains a sorted window of values for efficient statistical analysis.
        Inserts the new value in sorted order and removes the oldest value if
        the window exceeds the specified size.
        """
        if len(self.values) == len(self.sorted_window) + 1:
            val = self.values[-1]
            pos = self._binary_search(val)
            self.sorted_window.insert(pos, val)  # Insert in sorted order

            # Remove oldest value if window size exceeded
            if len(self.sorted_window) > self.window_size:
                old_val = self.values[-self.window_size - 1]
                pos = self._binary_search(old_val)
                self.sorted_window.pop(pos)

    def _get_robust_stats(self) -> Tuple[float, float]:
        """
        Computes robust statistics: median and Interquartile Range (IQR).
        Returns:
            Tuple[float, float]: The median and IQR of the sorted window.
        """
        if not self.sorted_window:
            return 0, 1

        n = len(self.sorted_window)
        median = self.sorted_window[n // 2]
        q1 = self.sorted_window[n // 4]
        q3 = self.sorted_window[3 * n // 4]
        iqr = q3 - q1

        return median, iqr if iqr > 0 else 1  # IQR fallback to prevent division by zero

    def _binary_search(self, value: float) -> int:
        """
        Binary search helper to find the insertion point for a new value in sorted_window.
        Args:
            value (float): The value to insert.
        Returns:
            int: The index where the value should be inserted.
        """
        left, right = 0, len(self.sorted_window)
        while left < right:
            mid = (left + right) // 2
            if self.sorted_window[mid] < value:
                left = mid + 1
            else:
                right = mid
        return left

    def update(self, value: float) -> Tuple[bool, float]:
        """
        Updates the detector with a new data point and checks if it's an anomaly.
        Args:
            value (float): The new data point value.
        Returns:
            Tuple[bool, float]: Boolean indicating if it's an anomaly, and the anomaly score.
        """
        try:
            # Add new value to the window
            self.values.append(value)

            # Update EMA and its standard deviation
            if self.ema is None:
                self.ema = value
                self.emstd = 0
            else:
                self.ema = self.alpha * value + (1 - self.alpha) * self.ema
                self.emstd = (
                    self.alpha * abs(value - self.ema) + (1 - self.alpha) * self.emstd
                )

            # Early detection during warm-up period
            if len(self.values) < self.window_size:
                if len(self.values) >= 5:  # Minimum points needed
                    mean = np.mean(list(self.values))
                    std = np.std(list(self.values))
                    score = abs(value - mean) / (std if std > 0 else 1)
                    # Use a slightly higher threshold during warm-up
                    warm_up_threshold = 3.5
                    return score > warm_up_threshold, score
                return False, 0.0

            # Update sorted window if window is full
            if len(self.values) >= self.window_size:
                self._update_sorted_window()

            # Calculate anomaly score based on robust stats and EMA
            median, iqr = self._get_robust_stats()
            static_score = abs(value - median) / (iqr * 0.7413)
            ema_score = abs(value - self.ema) / (self.emstd if self.emstd > 0 else 1)

            # Weighted score combining static and EMA-based scores
            score = 0.7 * static_score + 0.3 * ema_score

            # Dynamic threshold based on recent data distribution
            threshold = max(
                3.0,
                np.percentile(
                    [abs(v - median) / (iqr * 0.7413) for v in list(self.values)[-20:]],
                    95,
                ),
            )

            return score > threshold, score

        except Exception as e:
            print(f"Error processing value: {e}")
            return False, 0.0


class DataStreamSimulator:
    """
    Simulates a continuous data stream with periodic patterns, drift, and noise.
    """

    def __init__(self):
        self.t = 0  # Time step
        self.max_time = 1e6  # Max value to prevent overflow
        self.reset_threshold = -1e6  # Reset threshold

    def generate_point(self) -> Tuple[float, bool]:
        """
        Generates a data point with seasonal patterns, drift, and random noise.
        Returns:
            Tuple[float, bool]: The generated value and a boolean flag for anomaly.
        """
        try:
            if not isinstance(self.t, (int, float)):
                print("Invalid time value detected, resetting to 0")
                self.t = 0

            if self.t > self.max_time or self.t < self.reset_threshold:
                print(f"Time value {self.t} out of bounds, resetting to 0")
                self.t = 0

            # Components for synthetic data generation
            fast = 0.5 * np.sin(2 * np.pi * self.t / 50)  # Fast oscillation
            slow = 0.2 * np.sin(2 * np.pi * self.t / 200)  # Slow seasonal pattern
            drift = self.t * 0.001  # Gradual trend

            random_walk = np.sin(2 * np.pi * self.t / 100) * np.random.normal(0, 0.1)
            noise = np.random.normal(0, 0.2)

            value = fast + slow + drift + random_walk + noise

            if not np.isfinite(value):
                print("Invalid value generated, returning default")
                return 0.0, False

            is_anomaly = False
            if np.random.random() < 0.02:
                value += np.random.choice([-3, 3])
                is_anomaly = True

            self.t += 1
            return value, is_anomaly

        except Exception as e:
            print(f"Error in data generation: {e}")
            return 0.0, False


class StreamVisualizer:
    """
    Real-time visualizer for data streams and anomaly detection scores.

    This class provides a real-time, dual-panel visualization of a data stream
    and its corresponding anomaly scores. It plots the data stream on the top panel,
    highlighting detected anomalies with a red marker. The bottom panel shows the
    anomaly scores, with a threshold line indicating the anomaly detection limit.

    Attributes:
        fig (Figure): The Matplotlib figure object containing the subplots.
        ax1 (Axes): Top subplot for visualizing the data stream.
        ax2 (Axes): Bottom subplot for visualizing anomaly scores.
        values (List[float]): Stores the values of the data stream for plotting.
        scores (List[float]): Stores the anomaly scores corresponding to the data stream.
        anomalies_x (List[int]): X-coordinates of detected anomalies in the data stream.
        anomalies_y (List[float]): Y-values of detected anomalies in the data stream.

    Methods:
        update(point: DataPoint): Updates the visualization with a new data point.
            Appends the new data point to the stream, calculates and plots the
            anomaly score, and updates the plot in real-time.
    """

    def __init__(self):
        """
        Initializes the StreamVisualizer with two subplots for real-time visualization.

        - The top plot displays the data stream with a highlighted marker for anomalies.
        - The bottom plot shows the corresponding anomaly scores, with a threshold line.

        """
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.values = []
        self.scores = []
        self.anomalies_x = []
        self.anomalies_y = []

        # Setup axes with better styling
        self.ax1.set_ylim(-4, 4)
        self.ax1.set_title("Real-time Data Stream with Anomalies", fontsize=12)
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_ylim(0, 10)
        self.ax2.set_title("Anomaly Detection Scores", fontsize=12)
        self.ax2.grid(True, alpha=0.3)

    def update(self, point: DataPoint):
        """
        Updates the visualization with a new data point.
        Appends the new data point to the stream, calculates and plots the
        anomaly score, and updates the plot in real-time.
        """
        self.values.append(point.value)
        self.scores.append(point.anomaly_score)
        if point.is_anomaly:
            self.anomalies_x.append(len(self.values) - 1)
            self.anomalies_y.append(point.value)

        # Keep fixed window
        if len(self.values) > 100:
            self.values = self.values[-100:]
            self.scores = self.scores[-100:]
            self.anomalies_x = [
                x for x in self.anomalies_x if x >= len(self.values) - 100
            ]
            self.anomalies_y = self.anomalies_y[-len(self.anomalies_x) :]

        # Clear and redraw
        self.ax1.clear()
        self.ax2.clear()

        # Plot data stream with larger anomaly markers
        self.ax1.plot(self.values, "b-", label="Data Stream", linewidth=1.5)
        if self.anomalies_x:
            self.ax1.scatter(
                self.anomalies_x,
                self.anomalies_y,
                c="red",
                s=150,
                label="Anomalies",
                zorder=5,
                marker="o",
            )
        self.ax1.set_ylim(-4, 4)
        self.ax1.set_title("Real-time Data Stream with Anomalies", fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()

        # Plot scores with filled area under curve
        self.ax2.plot(self.scores, "g-", label="Anomaly Score", linewidth=1.5)
        self.ax2.fill_between(
            range(len(self.scores)), self.scores, alpha=0.2, color="green"
        )
        self.ax2.axhline(
            y=3.0, color="r", linestyle="--", label="Threshold", linewidth=1.5
        )
        self.ax2.set_ylim(0, max(10, max(self.scores) + 1))
        self.ax2.set_title("Anomaly Detection Scores", fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)


# Main script for testing the Anomaly Detection system.
if __name__ == "__main__":
    print("Starting sample data stream anomaly detection...")

    # Initialize components
    simulator = DataStreamSimulator()
    detector = AnomalyDetector(window_size=50, alpha=0.1)
    visualizer = StreamVisualizer()

    num_points = 100  # Number of data points to simulate

    # Run the simulation and detection on a sample data stream
    for _ in range(num_points):
        # Generate a new data point from the simulator
        value, true_anomaly = simulator.generate_point()

        # Detect anomaly using the detector
        is_anomaly, score = detector.update(value)

        # Create a data point object with the anomaly detection results
        point = DataPoint(
            value=value,
            timestamp=time.time(),
            is_anomaly=is_anomaly,
            anomaly_score=score,
        )

        # Update visualizer with the new data point
        visualizer.update(point)

        # Print detected anomalies for the sample run
        if is_anomaly:
            print(f"Anomaly detected! Value: {value:.2f}, Score: {score:.2f}")

        # Pause briefly for real-time effect in the visualization
        time.sleep(0.05)

    print("Sample data stream processing completed.")
