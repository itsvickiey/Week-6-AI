# 🌿 Edge AI Recyclable Classification Prototype

A lightweight, real-time recyclable item classifier powered by TensorFlow Lite.

## 📌 Overview

This project demonstrates how Edge AI can be used to classify recyclable materials directly on low-power devices such as Raspberry Pi. By training a compact CNN model and converting it to TensorFlow Lite, the system delivers real-time inference, low latency, and efficient on-device computation—ideal for smart recycling bins and environmental IoT solutions.

## 🎯 Objectives

Develop a lightweight CNN model for image classification

Convert the trained model into TensorFlow Lite format

Test performance differences between TensorFlow and TFLite

Simulate deployment on an edge device (Raspberry Pi)

Analyze speed, accuracy, and real-time inference capabilities

## 🛠️ Tech Stack

Frameworks: TensorFlow, TensorFlow Lite

Language: Python 3.8+

Environment: Google Colab / Local Machine

Hardware Target: Raspberry Pi (simulated)

Libraries: NumPy, Matplotlib, PIL, scikit-learn

## 📁 Project Structure
edge-ai-recyclable-classifier/
│
├── recyclable_classifier.py       # Model training & TFLite conversion
├── tflite_testing.py              # Accuracy comparison & inference tests
├── raspberry_pi_simulation.py     # Edge deployment simulation
├── performance_report.py          # Metrics visualization & report generation
├── recyclable_classifier.tflite   # Exported TFLite model
└── README.md                      # Documentation

## 🚀 Getting Started
1️⃣ Install Dependencies
pip install tensorflow tensorflow-datasets numpy matplotlib pillow scikit-learn

2️⃣ Train and Convert Model
from recyclable_classifier import main
classifier, history, test_accuracy, test_loss = main()

3️⃣ Evaluate TFLite Model
from tflite_testing import TFLiteTester

tester = TFLiteTester('recyclable_classifier.tflite')
tflite_accuracy, avg_inf_time, fps = tester.test_batch_inference(test_data, 100)

4️⃣ Simulate Edge Deployment
from raspberry_pi_simulation import main
main()

5️⃣ Generate Performance Report
from performance_report import generate_performance_report
metrics = generate_performance_report(history, test_accuracy, tflite_accuracy, avg_inf_time)

## 🧠 Model Architecture

A compact CNN optimized for edge inference:

Input (32×32×3)
↓
Conv2D (32 filters) + ReLU
↓
MaxPooling
↓
Conv2D (64 filters) + ReLU
↓
MaxPooling
↓
Conv2D (64 filters) + ReLU
↓
Flatten
↓
Dense (64) + ReLU
↓
Dropout (0.5)
↓
Softmax (10 classes)

## 🎮 Example Usage
Real-Time Classification Demo
from raspberry_pi_simulation import EdgeAISimulator

simulator = EdgeAISimulator('recyclable_classifier.tflite')
simulator.run_real_time_demo(duration=30)

Compare Original vs TFLite Model
from tflite_testing import compare_models

compare_models(classifier.model, 'recyclable_classifier.tflite', test_data)

## 📈 Performance Summary
Metric	Expected Value
Test Accuracy	70–85%
TFLite Accuracy	68–84%
Model Size	< 2 MB
Avg Inference Time	< 50 ms (Raspberry Pi)
Real-Time FPS	> 15
Conversion Loss	< 2%
## 🔧 Optimizations

FP16 Quantization — reduces model size

Pruning — improves inference speed

Hardware Acceleration — Edge TPU ready

Low Memory Footprint — suitable for IoT devices

## 🌟 Benefits of Edge AI

Low Latency: Real-time classification without cloud delay

Privacy: No sensitive data sent over the network

Resilience: Works offline, independent of connectivity

Cost Savings: Reduced cloud compute usage

Bandwidth Efficient: Only key results transmitted

## 🧩 Customization Guide
Use Your Own Dataset
def load_custom_dataset(data_path):
    return train_data, test_data

Customize Model Architecture
def build_custom_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        # Add more layers as needed...
    ])
    return model

📜 License

This project is open-source and available for educational and research use.
