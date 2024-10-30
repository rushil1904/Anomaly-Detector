"""test_anomaly_detection.py - Test suite for anomaly detection system

This script provides functions to:
1. Test the performance of the anomaly detection system (processing time, speed, and memory usage).
2. Evaluate accuracy through precision and recall.
3. Visualize real-time anomaly detection in a simulated data stream.

Dependencies:
- numpy
- matplotlib
- psutil (optional, for memory tracking)

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from anomaly_detection import (
    DataStreamSimulator,
    AnomalyDetector,
    DataPoint,
    StreamVisualizer,
)


def test_performance(n_points=10000):
    """
    Measures the performance of the anomaly detection system.

    Args:
        n_points (int): The number of data points to generate and analyze.

    Details:
        - Tracks data generation and detection times separately.
        - Calculates overall processing speed (points/second).
        - Tracks memory usage (if psutil is available) for insight into the memory footprint.
    """
    simulator = DataStreamSimulator()
    detector = AnomalyDetector()

    # Timing variables
    gen_time = 0  # Total time for data generation
    detect_time = 0  # Total time for anomaly detection
    total_start = time.time()  # Start time for the entire process
    anomaly_count = 0  # Total detected anomalies
    true_anomaly_count = 0  # Total true anomalies generated

    # Memory tracking setup (optional)
    initial_memory = 0
    peak_memory = 0
    try:
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
    except ImportError:
        print("psutil not available - memory tracking disabled")

    # Main loop for generating and analyzing data points
    for i in range(n_points):
        # Time the data generation step
        gen_start = time.time()
        value, true_anomaly = simulator.generate_point()
        gen_time += time.time() - gen_start

        # Time the anomaly detection step
        detect_start = time.time()
        is_anomaly, score = detector.update(value)
        detect_time += time.time() - detect_start

        # Track anomalies
        if true_anomaly:
            true_anomaly_count += 1
        if is_anomaly:
            anomaly_count += 1

        # Memory usage tracking (if available)
        try:
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
        except:
            pass

    # Calculate overall performance metrics
    total_time = time.time() - total_start
    points_per_second = n_points / total_time

    # Print performance and accuracy metrics
    print("\nDetailed Performance Results:")
    print(f"Total processing time: {total_time:.3f} seconds")
    print(
        f"Data generation time: {gen_time:.3f} seconds ({gen_time/total_time*100:.1f}%)"
    )
    print(
        f"Detection time: {detect_time:.3f} seconds ({detect_time/total_time*100:.1f}%)"
    )
    print(f"Overall speed: {points_per_second:.2f} points/second")

    print(f"\nAccuracy Metrics:")
    print(
        f"True anomalies: {true_anomaly_count} ({true_anomaly_count/n_points*100:.2f}%)"
    )
    print(f"Detected anomalies: {anomaly_count} ({anomaly_count/n_points*100:.2f}%)")

    # Print memory usage if available
    if initial_memory > 0:
        print(f"\nMemory Usage:")
        print(f"Initial: {initial_memory:.1f} MB")
        print(f"Peak: {peak_memory:.1f} MB")
        print(f"Increase: {peak_memory - initial_memory:.1f} MB")


def test_accuracy(n_points=1000):
    """
    Evaluates the accuracy of the anomaly detection system using precision and recall.

    Args:
        n_points (int): The number of data points to generate and analyze.

    Details:
        - Calculates true positives, false positives, and false negatives.
        - Prints precision and recall as accuracy metrics.
    """
    simulator = DataStreamSimulator()
    detector = AnomalyDetector()

    true_positives = 0  # Correctly identified anomalies
    false_positives = 0  # Non-anomalies incorrectly identified as anomalies
    false_negatives = 0  # Anomalies missed by the detector

    # Loop through each generated data point
    for _ in range(n_points):
        value, true_anomaly = simulator.generate_point()
        is_anomaly, score = detector.update(value)

        if true_anomaly and is_anomaly:
            true_positives += 1
        elif not true_anomaly and is_anomaly:
            false_positives += 1
        elif true_anomaly and not is_anomaly:
            false_negatives += 1

    # Display accuracy metrics
    print("\nAccuracy Test Results:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    # Calculate and print recall and precision
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Detection Rate: {recall*100:.1f}%")
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision: {precision*100:.1f}%")


def visualize_detection(duration_seconds=10):
    """
    Visualizes real-time anomaly detection over a simulated data stream.
    """
    simulator = DataStreamSimulator()
    detector = AnomalyDetector(window_size=50, alpha=0.1)
    visualizer = StreamVisualizer()

    anomalies_detected = 0
    start_time = time.time()
    print(f"\nStarting {duration_seconds}-second visualization...")

    try:
        while time.time() - start_time < duration_seconds:
            # Generate and detect
            value, true_anomaly = simulator.generate_point()
            is_anomaly, score = detector.update(value)

            # Create data point
            point = DataPoint(
                value=value,
                timestamp=time.time(),
                is_anomaly=is_anomaly,
                anomaly_score=score,
            )

            # Update visualization
            visualizer.update(point)

            if is_anomaly:
                anomalies_detected += 1
                print(
                    f"Anomaly {anomalies_detected} detected: Value={value:.2f}, Score={score:.2f}"
                )

            time.sleep(0.05)

        # Save final state with both plots
        plt.gcf().set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig("anomaly_detection_test.png", dpi=300, bbox_inches="tight")
        print(f"\nVisualization saved as 'anomaly_detection_test.png'")
        print(f"Total anomalies detected: {anomalies_detected}")

    except KeyboardInterrupt:
        print("\nVisualization terminated by user")
    finally:
        plt.close()

    return anomalies_detected


if __name__ == "__main__":
    print("Running comprehensive tests...")

    # Run performance test
    test_performance()

    # Run accuracy test
    test_accuracy()

    # Run real-time visualization for 10 seconds
    print("\nStarting visualization test (10 seconds)...")
    visualize_detection()
