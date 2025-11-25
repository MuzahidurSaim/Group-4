# Department of Electrical and Computer Engineering
**CSE 299: Junior Design Project, Sec – 6, Group - 4**  
**Basic Project Proposal – Fall 2025**

---

## 1. Title of the Project
**Plant Disease Detection Mobile Application Using Deep Learning (CNN)**

---

## 2. Introduction

### Project Purpose
This project aims to develop a mobile application that leverages deep learning, specifically Convolutional Neural Networks (CNN), to detect plant diseases from leaf images. Users such as farmers and gardeners will be able to capture or upload a photo of a plant leaf and instantly receive information about potential diseases, along with suggested remedies.

### Motivation
Plant diseases cause significant crop losses every year, directly impacting food supply and farmers’ income. Many farmers, especially in rural areas, lack access to agricultural experts for timely diagnosis. With the increasing use of smartphones, a mobile-based intelligent system can provide quick, affordable, and accessible disease detection.

### Problem Statement
Farmers and gardeners often struggle to identify plant diseases accurately due to lack of expertise. Manual inspection is time-consuming and subjective, and the absence of accessible diagnostic tools leads to delayed treatment and reduced crop yield.

### Existing Limitations
Current plant disease detection apps are often limited to a single crop or disease, have complex interfaces, or lack offline functionality. Our proposed app will support multiple plant types, provide higher accuracy, and feature a simple, user-friendly interface suitable for all users.

---

## 3. Functional Requirements

### User and Expert Interviews
Discussions with home gardeners and agricultural experts highlighted the need for:
- A quick, image-based detection system
- Easy navigation without technical knowledge
- High detection accuracy and reliable dataset training

**Interview Data Reference:**  
Voice recordings and transcripts of interviews are stored in the project’s shared Google Drive [folder](https://drive.google.com/drive/folders/13VrSsz-FUCujly5by3zUCHVaaXeCEtCO?usp=drive_link): *Interview 
Data & Voice Records*

### System Functionalities
The application will provide:
- Capture or upload a photo of the plant leaf
- Detect disease using a pre-trained CNN model
- Display disease name and confidence level
- Work offline using a built-in TensorFlow Lite model
- Save recent predictions for later reference
- Send reminder notifications for crop health checks

---

## 4. Non-Functional Requirements
- **Fast:** Results should appear within 3 seconds
- **Lightweight:** App size kept below 100 MB
- **Usable:** Simple interface with clear buttons and short texts
- **Secure:** Images remain on the device; no external sharing
- **Stable:** Runs smoothly on mid-range Android devices
- **Updatable:** Model and content can be updated without redesigning the app

---

## 5. Technology Stack
- **Frontend:** Android Studio (Kotlin/Java) or Flutter for cross-platform support
- **Model Integration:** TensorFlow Lite (converted from CNN)
- **Database:** SQLite for local storage of predictions
- **Optional Backend:** Flask or FastAPI for future online predictions
- **Version Control:** Git and GitHub
- **Deployment:** Google Play Store (internal testing before release)

---

## 6. Project Plan and Timeline

| Phase    | Duration                               | Tasks                                                      |
|----------|----------------------------------------|------------------------------------------------------------|
| Week 1–2 | Requirements Gathering & System Design | Identify user needs, design architecture, prepare datasets |
| Week 3–4 | Model Development                      | Train and test CNN model using PlantVillage dataset        |
| Week 5–6 | Mobile App Development                 | Build user interface and integrate basic functionalities   |
| Week 7–8 | Model Integration & Optimization       | Convert model to TensorFlow Lite, test on mobile           |
| Week 9   | Testing & Debugging                    | Perform functional and performance testing                 |
| Week 10  | Deployment & Final Presentation        | Deploy app, prepare documentation, and present             |

---

## 7. Comparison with Existing Apps

| Feature / Criteria         | Existing Apps (Plantix, Leaf Doctor) | Proposed App (Our Project)                   |
|----------------------------|--------------------------------------|----------------------------------------------|
| **Crop Coverage**          | Often limited to specific crops      | Supports multiple plant types                |
| **Disease Coverage**       | Restricted to a few common diseases  | Broader coverage, extendable                 |
| **Interface Usability**    | Complex navigation, technical terms  | Simple, farmer-friendly UI                   |
| **Offline Functionality**  | Requires internet connection         | Works offline with TensorFlow Lite           |
| **Accuracy**               | Varies (70–85%)                      | Targeting higher accuracy with optimized CNN |
| **Data Privacy**           | Images uploaded to cloud             | Images processed locally                     |
| **Notifications**          | Rarely included                      | Built-in reminders                           |
| **App Size & Performance** | Can be heavy (>150 MB)               | Lightweight (<100 MB)                        |

---

## 8. Team Structure
- **Muzahidur Rahman Saim** – Machine Learning and Backend Developer
- **Md Abid Hossain** – Mobile App Developer
- **Monia Nazmul Prity** – Database Manager and Tester

---

## 9. Conclusion
This project will deliver a reliable and accessible mobile application for plant disease detection using deep learning. By combining CNN-based image recognition with a lightweight, user-friendly mobile interface, the system will empower farmers and gardeners with instant, accurate disease diagnosis. This solution reduces dependency on experts, prevents crop loss, and supports the broader vision of smart and sustainable agriculture.

---

## 10. References
- Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060.
- Plantix. (n.d.). Retrieved from [https://plantix.net](https://plantix.net)
- Leaf Doctor. (n.d.). Retrieved from [https://apps.apple.com/us/app/leaf-doctor/id1003260350](https://apps.apple.com/us/app/leaf-doctor/id1003260350)
- TensorFlow Lite. (n.d.). Retrieved from [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
- Interview Data & Voice Records. (2025). Project team’s primary research. Available at: *Google Drive Link*  

---

## 11. Model Training Data

### Model Info

- Two deployable checkpoints live in `Group-4/saved_models`: `latest_model.h5` (170 MB) and `model_epoch_61.keras` (170 MB). `class_names.json` is a lightweight 179 B label map; these file sizes come from `ls -lh`.

- Loading `latest_model.h5` in TensorFlow shows a 3-block CNN with 14,840,008 trainable parameters (≈56.6 MB of weights). That summary also reports the layer stack (input preprocessing → 3×Conv/MaxPool → Flatten → Dense 128 → Dense 8) so you can cite it if needed.

### Training Setup

- Core hyperparameters and dataset target are defined in `training/training.ipynb`: 256×256 RGB inputs, batch size 32, PlantVillage directory as the source, and the model artifacts are persisted to `../saved_models`.

- The notebook finds 6,627 labeled images across 8 disease/health classes, then splits them 80 %/10 %/10 % via `get_dataset_partitions_tf`. Each subset uses caching, shuffling, and `tf.data.AUTOTUNE` prefetching for throughput.

- Preprocessing is embedded in the model graph: resize/rescale plus augmentation (`RandomFlip` + `RandomRotation`), so the exported model can accept raw 256×256 images at inference.

- The convolutional stack is three `Conv2D` +` MaxPooling2D` blocks feeding a 128-unit dense head and an 8-way softmax classifier; no transfer learning backbone is used.

- Training uses Adam with `SparseCategoricalCrossentropy` and tracks accuracy.

- Checkpoints are written every epoch (`model_epoch_XX.keras`) and an `EarlyStopping` callback with patience 5 restores the best weights. The helper `resume_training` function later in the notebook supports fine-tuning or class-count changes by freezing/unfreezing layers and adjusting the final Dense layer before continuing.

---