# 🦾 DexGraspDetector: Physics-Informed Grasp Detection

🚀 Official Implementation of **"PHYSICS-INFORMED GRASP DETECTION: ATTENTION-DRIVEN FORCE-CLOSURE ANALYSIS FOR DEXTEROUS MANIPULATION"**  
📜 **Authors**:Zimo Wen 
🛠 **Code**: This repo provides the official implementation of our proposed grasp detection model.  

---

## 📌 Overview

Dexterous grasp detection is crucial for stable robotic manipulation, yet most methods rely on **RGB-D or depth images**, limiting their ability to utilize other **robotic state inputs**, such as **grasp poses and joint configurations**.

🔥 **We propose DexGraspDetector, a novel approach integrating:**
- **Geometric Learning** with PointNet++ for feature extraction
- **Force-Closure Analysis** for grasp stability
- **An Attention Mechanism** to balance multimodal inputs dynamically

📈 **Our model achieves 93.2% accuracy**, surpassing existing methods and demonstrating the advantages of integrating **physical stability constraints**.

---

## 📸 Model Architecture

Our framework consists of:
1. **Force-Closure Analysis**: Computes stability scores based on contact points and friction forces.
2. **Geometric Feature Extraction**: Uses PointNet++ to extract features from 3D point clouds.
3. **Multimodal Attention Fusion**: Dynamically balances **point cloud, pose, and joint data** for accurate grasp predictions.

---


