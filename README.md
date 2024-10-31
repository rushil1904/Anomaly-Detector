# Efficient Data Stream Anomaly Detection

![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview
This implementation provides a real-time anomaly detection system for continuous data streams using robust statistical methods. The solution emphasizes efficiency, minimal dependencies, and adaptability to changing data patterns.


## Key Components

### Core Module (`anomaly_detection.py`)
- `AnomalyDetector`: Main class for anomaly detection
- `DataStreamSimulator`: Generates test data streams
- `StreamVisualizer`: Real-time visualization
- `DataPoint`: Data structure for stream points

### Test Suite (`test_anomaly_detection.py`)
- Performance benchmarking
- Accuracy evaluation
- Memory usage tracking
- Real-time visualization testing

### Examples (`examples/simple_detection.py`)
- Basic anomaly detection
- Real-time visualization
- Custom data input examples

## Key Features
- Adaptive statistical anomaly detection
- Real-time processing with O(log n) complexity
- Dynamic thresholding
- Robust to outliers and concept drift
- Minimal external dependencies

## Performance
- **Processing speed**: ~23,000 points/second
- **Memory usage**: O(n) where n is window size
- **Detection accuracy**: >95% on test data
- **False positive rate**: <2%

## Dependencies

### Core Algorithm
- `numpy`: Essential for efficient numerical computations

### Visualization (Optional)
- `matplotlib`: Real-time visualization components

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/username/anomaly-detection.git
    cd anomaly-detection
    ```

2. Install dependencies:
    ```bash
    # Core only
    pip install numpy

    # Full installation (with visualization)
    pip install -r requirements.txt
    ```

## Usage

### Basic Anomaly Detection
```python
from anomaly_detection import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(window_size=100)

# Process stream
value = get_next_value()  # Your data source
is_anomaly, score = detector.update(value)

if is_anomaly:
    print(f"Anomaly detected! Score: {score:.2f}")
```

## Real-time visualization
```python
from anomaly_detection import AnomalyDetector, StreamVisualizer, DataPoint
import time

# Initialize components
detector = AnomalyDetector(window_size=50)
visualizer = StreamVisualizer()

# Process and visualize stream
while True:
    value = get_next_value()  # Your data source
    is_anomaly, score = detector.update(value)
    
    point = DataPoint(
        value=value,
        timestamp=time.time(),
        is_anomaly=is_anomaly,
        anomaly_score=score
    )
    
    visualizer.update(point)
    time.sleep(0.05)  # Control update rate
```

## Running tests
```python
# Run all tests
python test_anomaly_detection.py

# Run examples
python examples/simple_detection.py
```

## Algorithm Details
The anomaly detector uses a combination of:

- Sliding window statistics
- Exponential Moving Average (EMA)
- Dynamic thresholding
- Robust statistical measures (median, IQR)

## Contributing
Contributions welcome!

## License
MIT License