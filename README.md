# Trustworthiness Evaluation: AI-Powered Traffic Management System

**Course:** CIS 6930 Trustworthy AI Systems  
**Group 24:** Suma Vatturi (U44064470), RamaKrishna Reddy Vennam (U85778186)  
**Midterm Project GitHub:** https://github.com/suma-vatturi/Trustness-Evaluation-Traffic-Management-System  

---

## 1 – Introduction

Our midterm built an AI-driven dynamic traffic signal controller using YOLOv9 to detect vehicles and adapt signal timings in real time. For this final project, we evaluated two key trustworthiness principles:

1. **Reliability & Robustness**  
> How well does our system maintain detection performance when confronted with real-world perturbations (lighting changes, weather, motion blur, etc.)?

2. **Fairness & Bias**
> How equitably does our system perform across different vehicle types, particularly for vulnerable road users?

We chose these aspects because:
- Traffic management systems must be **robust** to operate reliably 24/7 in varying conditions
- They must also be **fair** to ensure equitable service for all road users, from trucks to bicycles
- The combination of robustness and fairness is critical for real-world deployment safety

For our evaluation, instead of the augmented 8,693-image dataset and bleeding-edge YOLOv9 from midterm, we:

- Switched to **YOLOv8s**, which integrates smoothly with existing Python libraries and evaluation pipelines
- Used an **unaugmented** subset of **3,649** Street-View images to:
  - Avoid synthetic data artifacts in robustness testing
  - Better assess natural class imbalances for fairness evaluation
  - Provide a more realistic assessment of real-world performance

This README walks through our entire workflow—from training through robustness and fairness analyses—demonstrating how our traffic management system performs not just under various environmental conditions, but also across different vehicle classes and road user types.

Our evaluation revealed important insights about:
- System resilience to environmental challenges
- Detection bias between common and rare vehicle classes
- Performance gaps affecting vulnerable road users
- Critical scenarios requiring additional safety measures

These findings are essential for deploying a traffic management system that is both reliable and equitable.

---

## 2 – Trustworthiness Objectives

For our evaluation, we focused on two critical aspects of trustworthiness:

### 2.1 Reliability & Robustness
> How well does our system maintain detection performance when confronted with real‐world perturbations?

Because traffic management systems must operate reliably 24/7 in varying conditions, we conducted:

1. **Comprehensive Environmental Testing**:
   - **Lighting Variations**: Low light (night) and bright glare (sun)
   - **Weather Conditions**: Fog, rain, and snow simulation
   - **Motion Effects**: Blur simulation for high-speed scenarios

2. **Multi-Metric Performance Analysis**:
   - **Detection Stability**: Average detection count across conditions
   - **Accuracy Metrics**: Precision, Recall, F1-score per condition
   - **Confidence Analysis**: Average confidence and IoU tracking
   - **Error Patterns**: False positive/negative rates per image

3. **Systematic Evaluation Process**:
   - Baseline establishment in optimal conditions
   - Progressive stress testing with increasing perturbation severity
   - Statistical significance analysis of performance drops
   - Identification of critical failure modes

### 2.2 Fairness & Bias
> How equitable is our system's performance across different vehicle types and road users?

Because traffic management must serve all road users fairly, we conducted:

1. **Multi-Class Performance Analysis**:
   - **Vehicle Size Impact**: Large (trucks/buses) vs small (bikes) detection rates
   - **Frequency Bias**: Common (cars) vs rare (motorcycles) class performance
   - **Vulnerable Users**: Special focus on pedestrian and bicycle detection
   - **Environmental Impact**: Class-specific degradation in adverse conditions

2. **Fairness Metrics Evaluation**:
   - **Per-Class Statistics**: Precision, Recall, F1-score breakdowns
   - **Detection Rate Equity**: True Positive rates across classes
   - **Error Distribution**: False positive/negative patterns by class
   - **Confusion Analysis**: Inter-class misclassification study

3. **Bias Identification & Mitigation**:
   - Quantification of performance gaps between classes
   - Analysis of systematic classification errors
   - Investigation of size-based detection bias
   - Recommendations for balanced model improvement

### 2.3 Integration of Objectives

Our dual focus on robustness and fairness provides a comprehensive evaluation because:
1. **Real-World Applicability**: Systems must be both robust AND fair to be truly trustworthy
2. **Safety Critical**: Reliable detection of ALL road users is essential for traffic safety
3. **Operational Equity**: Performance should not degrade disproportionately for any user group
4. **Deployment Readiness**: Understanding both aspects helps inform practical deployment decisions

This evaluation framework helps ensure our traffic management system is:
- **Reliable**: Maintains performance across environmental conditions
- **Robust**: Degrades gracefully under stress
- **Fair**: Serves all road users equitably
- **Safe**: Prioritizes detection of vulnerable users
- **Practical**: Ready for real-world deployment challenges

Our results provide actionable insights for:
- Model improvement priorities
- Deployment environment considerations
- Safety margin requirements
- Fairness enhancement strategies

---

## 3 – Repository Structure

```
Trustness-Evaluation-Traffic-Management-System/
│
├── README.md                        ← this file
├── requirements.txt                 ← all Python dependencies
├── data/                            ← data configuration & class map
│   ├── data.yaml
│   └── class_map.json
│
├── models/                          ← trained weights
│   └── best.pt
│
├── notebooks/                       ← Jupyter workflows
│   ├── 01_training.ipynb
│   ├── 02_robustness.ipynb
│   └── 03_fairness.ipynb
│
├── src/                             ← script entrypoints
│   ├── train/                       ← training code
│   │   └── train.py
│   └── evaluate/                    ← evaluation code
│       ├── robustness.py
│       └── fairness.py
│
├── results/                         ← example output plots & tables
│   ├── robustness/
│   └── fairness/
│
└── src/app/                         ← Streamlit web demo
    └── app.py
```

---

## 4 – Setup & Installation

1. **Clone** the repo:
   ```bash
   git clone https://github.com/suma-vatturi/Trustness-Evaluation-Traffic-Management-System.git
   cd Trustness-Evaluation-Traffic-Management-System
   ```
2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare** dataset:  
   - If you already have `data/Street-View-1/` locally, skip downloading.  
   - Otherwise run the Roboflow download cell in `01_training.ipynb`.

---

## 5 – Training the Model

### 5.1 Notebook: `01_training.ipynb`

- **Why YOLOv8s & unaugmented data?**  
  YOLOv8s is stable, reproducible, and plays nicely with our evaluation code. An unaugmented core dataset (3,649 images) prevents artificial boosts from synthetic transforms, so our robustness tests reflect real-world gaps.

- **Key steps**:
  1. Check GPU availability using `nvidia-smi`
  2. Install required packages:
     - `ultralytics==8.2.103` (YOLOv8)
     - `roboflow==1.1.48` (for dataset management)
  3. Download and prepare dataset:
     - Use Roboflow API to fetch Street-View dataset
     - Dataset version 1 in YOLOv9 format
  4. Train YOLOv8s model:
     ```bash
     yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=640 batch=16 plots=True
     ```
  5. Review training results:
     - Confusion matrix
     - Training metrics graphs
     - Validation batch predictions
  6. Validate trained model:
     ```bash
     yolo task=detect mode=val model=runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
     ```
  7. Run inference with custom model:
     - Predict on test images with confidence threshold 0.25
     - Save and display prediction results

---

## 6 – Robustness Evaluation

### 6.1 Notebook: `02_robustness.ipynb`

- **Approach**: Apply six real-world perturbations to each test image, then run `model.predict()` to avoid DataLoader/NumPy cache issues.
- **Metrics**:
  - **Avg detections per image**  
  - **Precision / Recall / F1 / FP_per_image / FN_per_image**  
  - **Avg confidence** and **Avg IoU** of true positives  
  - **Normalized confusion matrix** (rows sum to 1.0)  
  - **AP@0.5:0.95** proxy via IoU sweep

**How to run**:
```bash
python src/evaluate/robustness.py   --model models/best.pt   --data data/data.yaml   --input_datasets data/Street-View-1/test/images   --output_folder results/robustness
```
Or open `notebooks/02_robustness.ipynb` and run top to bottom.

---

## 7 – Fairness Evaluation

### 7.1 Notebook: `03_fairness.ipynb`

- **Goal**: Measure per-class detection quality, exposing bias between common (cars) and rare (bikes) classes.
- **Per-class metrics**:
  - **Precision**, **Recall**, **F1-score**, **Detection rate (TP/GT)**
  - **AvgConf**, **AvgIoU**, **FP/Image**, **FN/Image**
  - **Normalized confusion matrix**  
  - **Approximate AP@0.5:0.95** for each class

**How to run**:
```bash
python src/evaluate/fairness.py   --model models/best.pt   --images data/Street-View-1/test/images   --labels data/Street-View-1/test/labels   --output_folder results/fairness
```
Or launch `notebooks/03_fairness.ipynb`.

---

## 8 – Experimental Results

### 8.1 Robustness Results

| Condition     | AvgDetections | Precision | Recall | F1-Score | AvgConf | AvgIoU |
|--------------:|--------------:|----------:|-------:|---------:|--------:|-------:|
| original      |         12.30 |     0.91  |  0.90  |    0.90  |   0.85  |  0.74  |
| low_light     |         12.04 |     0.90  |  0.88  |    0.89  |   0.84  |  0.72  |
| bright_light  |         11.46 |     0.88  |  0.85  |    0.86  |   0.82  |  0.70  |
| fog           |          6.77 |     0.78  |  0.65  |    0.71  |   0.76  |  0.68  |
| rain          |         12.12 |     0.89  |  0.89  |    0.89  |   0.85  |  0.73  |
| snow          |          3.60 |     0.70  |  0.50  |    0.58  |   0.72  |  0.65  |
| motion_blur   |          3.34 |     0.68  |  0.45  |    0.54  |   0.70  |  0.60  |

Key Findings:
1. **Strong Baseline Performance**:
   - Model achieves 90% F1-score under optimal conditions
   - Maintains high precision (0.91) and recall (0.90)
   - Good localization with IoU of 0.74

2. **Resilience to Common Conditions**:
   - Low light: Only 2% drop in detection count (12.04 vs 12.30)
   - Rain: Maintains 89% F1-score, nearly matching baseline
   - Bright light: 86% F1-score with good confidence (0.82)

3. **Critical Challenges**:
   - Snow: 71% drop in detection count (3.60 vs 12.30)
   - Motion blur: Worst performance with only 3.34 detections/image
   - Fog: Significant impact with 45% detection drop

### 8.2 Fairness Results

| Class      | Precision | Recall | F1-Score | Det_Rate |
|:-----------|----------:|-------:|---------:|---------:|
| bicycle    |     0.77  |  0.83  |    0.80  |    0.90  |
| bus        |     0.86  |  0.88  |    0.87  |    0.95  |
| car        |     0.95  |  0.93  |    0.94  |    0.94  |
| motorbike  |     0.79  |  0.85  |    0.82  |    0.87  |
| person     |     0.89  |  0.85  |    0.87  |    0.87  |
| truck      |     0.87  |  0.91  |    0.89  |    0.95  |

Key Observations:
1. **Class Performance Hierarchy**:
   - Cars: Best overall performance (F1=0.94)
   - Large vehicles (trucks/buses): Strong detection (0.87-0.89 F1)
   - Two-wheelers: Lower performance (0.80-0.82 F1)

2. **Fairness Gaps**:
   - 14% F1-score gap between best (cars: 0.94) and worst (bicycles: 0.80)
   - Two-wheelers consistently underperform by 10-15%
   - Person detection shows good balance (0.87 F1)

3. **Detection Rate Analysis**:
   - Strong for large vehicles (0.95 for buses/trucks)
   - Good for cars (0.94)
   - Lower for vulnerable road users (0.87-0.90)

These results indicate a well-performing system for standard conditions and common vehicle types, but with clear areas needing improvement for adverse weather and smaller road users.

---

## 9 – Citations & Acknowledgments

- **Ultralytics YOLOv8** (ultralytics v8.2.103)  
- **Roboflow Street-View Dataset**  
- Adapted ideas from YOLOv9 tutorial (Ultralytics, 2024)  
- Evaluation code inspired by open-source notebooks (links in scripts)

---

## 10 – Submission Checklist

- [x] **Public GitHub repo** with link submitted on Canvas  
- [x] Files are **organized** under `/src`, `/data`, `/notebooks`, `/results`  
- [x] **README.md** at repo root with clear instructions  
- [x] Dependencies listed in **requirements.txt**  
- [x] All external code and libraries properly **cited**  

Thank you for reviewing our work! 🚦🚗🛡️
