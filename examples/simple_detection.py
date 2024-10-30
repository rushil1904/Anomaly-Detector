"""
examples/simple_detection.py - Simple example of using the anomaly detection system.

This example demonstrates:
1. Basic anomaly detection
2. Real-time visualization
3. Custom data input
"""

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import matplotlib.pyplot as plt
from anomaly_detection import AnomalyDetector, DataPoint, StreamVisualizer


def basic_example():
    """Basic usage example without visualization."""
    print("\nRunning basic anomaly detection example...")

    # Initialize detector
    detector = AnomalyDetector(window_size=20, alpha=0.1)

    # Generate some sample data with an obvious anomaly
    normal_data = np.random.normal(0, 1, 30)  # 30 points of normal data
    data = list(normal_data)
    data.extend([10.0])  # Add an obvious anomaly
    data.extend(normal_data)  # More normal data

    # Process each point
    for i, value in enumerate(data):
        is_anomaly, score = detector.update(value)
        if is_anomaly:
            print(
                f"Anomaly detected at point {i}: value={value:.2f}, score={score:.2f}"
            )


def real_time_example():
    """Example with real-time visualization."""
    print("\nRunning real-time visualization example...")

    detector = AnomalyDetector(window_size=50, alpha=0.1)
    visualizer = StreamVisualizer()

    # Generate synthetic data stream
    t = 0
    anomalies_detected = 0
    try:
        for _ in range(100):  # Process 100 points
            # Generate value (sine wave + noise + occasional spike)
            value = np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.2)

            # Add occasional spikes
            if np.random.random() < 0.05:  # 5% chance of spike
                value += np.random.choice([-3, 3])

            # Detect anomalies
            is_anomaly, score = detector.update(value)

            # Create data point and update visualization
            point = DataPoint(
                value=value,
                timestamp=time.time(),
                is_anomaly=is_anomaly,
                anomaly_score=score,
            )

            visualizer.update(point)
            if is_anomaly:
                anomalies_detected += 1
                print(f"Anomaly detected! Value: {value:.2f}, Score: {score:.2f}")

            time.sleep(0.05)  # Short pause for visualization
            t += 1

        # Save final state
        plt.gcf().set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig("examples/example_detection.png", dpi=300, bbox_inches="tight")
        print(f"\nFinal visualization saved as 'examples/example_detection.png'")
        print(f"Total anomalies detected: {anomalies_detected}")

    except KeyboardInterrupt:
        print("\nVisualization terminated by user")
    finally:
        plt.close()


def custom_data_example():
    """Example using custom data input."""
    print("\nRunning custom data example...")

    detector = AnomalyDetector(window_size=10, alpha=0.1)

    # Example: monitoring system metrics
    cpu_utilization = [
        30,
        32,
        31,
        33,
        32,
        35,
        33,
        31,
        32,
        95,  # Sudden spike to 95%
        34,
        33,
        31,
        32,
        35,
        33,
        32,
        31,
        33,
        32,
    ]

    print("Monitoring CPU utilization for anomalies...")
    for i, cpu in enumerate(cpu_utilization):
        is_anomaly, score = detector.update(cpu)
        print(
            f"CPU: {cpu}% -> {'ANOMALY' if is_anomaly else 'normal'} (score: {score:.2f})"
        )


if __name__ == "__main__":
    print("Anomaly Detection Examples")
    print("-" * 50)

    # Run examples
    basic_example()
    custom_data_example()

    print("\nStarting real-time visualization (press Ctrl+C to stop)...")
    real_time_example()
