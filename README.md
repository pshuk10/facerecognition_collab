# Face Recognition Based Door Lock System

An intelligent, contactless, and affordable IoT-based door security system. This project integrates embedded hardware with deep learning-based cloud inference to provide a robust alternative to traditional authentication systems like RFID or PIN-based locks.

---

## 🚀 Overview

The system uses an **ESP32-CAM** module to capture facial images and transmit them to a **Python Flask server**. The server processes these images using the **Insight Face** framework to authenticate users. Upon successful recognition, a signal is sent back to the ESP32 to trigger a **solenoid lock** via a relay module.

### Key Features
* **Contactless Authentication:** Uses facial recognition for hygienic and convenient access.
* **Deep Learning Inference:** Utilizes the Insight Face Buffalo model for high accuracy.
* **Low Cost:** Built using affordable, open-source components.
* **Privacy-Focused:** Automated image cleanup via cron jobs ensures data is not stored longer than necessary.
* **Real-time Performance:** Average recognition latency of ~1.5 seconds.

---

## 🛠️ System Architecture

The architecture follows a modular approach, distributing tasks between the edge (ESP32) and the local cloud (Flask Server):

| Unit | Component | Function |
| :--- | :--- | :--- |
| **Input** | ESP32-CAM | Captures frames and provides Wi-Fi connectivity. |
| **Processing** | Flask Server | Performs heavy CNN inference for face detection and matching. |
| **Memory** | PSRAM / Server DB | Temporary frame buffering and authorized embedding storage. |
| **Control** | ESP32 GPIO | Validates server responses and issues hardware signals. |
| **Output** | Solenoid Lock | Physical actuation for opening/closing the door. |

---

## 🔌 Hardware Configuration

The hardware setup is powered by a 12V DC source, regulated for the microcontroller:
* **ESP32-CAM:** The central controller for image acquisition.
* **Solenoid Lock:** 12V electronic lock mechanism.
* **Relay Module:** Single-channel opto-isolated relay to interface between ESP32 and the lock.
* **Buck Converter:** Steps down 12V to 5V for the ESP32.
* **FTDI Adapter:** Used for initial programming and debugging.

---

## 💻 Software Design

### ESP32-CAM Firmware (Arduino C++)
* Initializes Camera and PSRAM.
* Captures images and compresses them into JPEG format.
* Transmits images via **HTTP multipart POST** requests.
* Manages a Finite State Machine (FSM) for IDLE, CAPTURE, VERIFY, and ACTUATE states.

### Flask Server (Python)
* Receives images at the `/upload` endpoint.
* Uses **Insight Face Buffalo ONNX** model for generating 512-D embeddings.
* Compares embeddings using **ArcFace angular similarity**.
* Implements **OpenBLAS** for multi-core computation.

---

## 📊 Performance Metrics

Testing under controlled indoor lighting yielded the following results:

* **Recognition Accuracy:** ≈ 96%
* **Average Latency:** ~1.5 seconds
* **False Acceptance Rate (FAR):** < 3%
* **False Rejection Rate (FRR):** < 5%

> **Note:** Performance may slightly drop in low-light or backlit conditions (Similarity scores typically drop below 0.5 at distances > 2.5m in very low light).

---

## 👥 Team Members
* Pranay Shukla
* Piyush Agarwal
* Arpit Kumar Gupta
* Aman Gupta
