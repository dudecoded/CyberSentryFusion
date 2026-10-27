# 🛡️ CyberSentryFusion

### Explainable AI-Powered Network Intrusion Detection

CyberSentryFusion is a machine-learning-based network intrusion detection system designed to analyze network traffic records and identify potentially malicious activity. It combines an **XGBoost classifier, SHAP-based explainability, configurable decision thresholds, and an interactive Streamlit dashboard** to make intrusion detection easier to explore and understand.

The project brings together data exploration, preprocessing, model evaluation, explainable AI, and interactive prediction in one unified application.

> **Project focus:** Network intrusion detection, machine learning, explainable AI (XAI), and interactive security analytics.

---

## 🚀 Live Demo

https://cybersentryfusion-cxk4fthgnbcfhwyqrbfq2r.streamlit.app/



---

## ✨ Key Features

### 📊 Interactive Security Dashboard

* Dark, security-focused interface.
* Visual summaries of prediction results.
* Interactive charts for exploring network traffic classifications.

### 🤖 Machine Learning Detection

* Uses a trained XGBoost classifier.
* Applies a saved preprocessing pipeline to incoming records.
* Uses a configurable classification threshold.
* Classifies records as `NORMAL` or `ATTACK`.

### 🔍 Explainable AI with SHAP

* Explores how input features influence model predictions.
* Helps users investigate the factors contributing to a prediction.
* Makes model behaviour easier to interpret beyond the predicted label.

### 🧪 Interactive Traffic Simulator

* Experiment with predefined traffic scenarios.
* Modify input features and observe prediction changes.
* Compare multiple simulated scenarios using visualizations.

### 📁 CSV Batch Analysis

* Upload a CSV containing network traffic records.
* Process multiple records in a single operation.
* Review predictions and associated model scores.
* Download prediction reports for further analysis.

### 📈 Model Evaluation and Research

The project includes organized Jupyter notebooks covering:

* Dataset exploration and analysis.
* Data preprocessing and model training.
* Model explainability.
* Model testing and evaluation.

### 📝 Prediction History

* Review predictions generated during the current application session.
* Compare previously analyzed records.
* Use available report downloads to retain analysis results.

---

## 🧠 Machine Learning Pipeline

CyberSentryFusion follows a typical supervised machine-learning workflow.

```text
UNSW-NB15 Dataset
        │
        ▼
Data Exploration
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Transformation
        │
        ▼
Trained XGBoost Classifier
        │
        ▼
Classification Score
        │
        ▼
Decision Threshold
        │
        ▼
NORMAL / ATTACK
        │
        ▼
Dashboard, Visualizations
and Prediction Reports
```

SHAP explainability provides an additional way to investigate the model's predictions.

### Prediction Workflow

1. The user supplies traffic records through the simulator or CSV uploader.
2. The saved preprocessing pipeline transforms the input features.
3. The trained XGBoost classifier calculates prediction scores.
4. The configured threshold determines the predicted class.
5. The dashboard displays the results and available visual explanations.

**Important:** The model identifies patterns associated with attacks in its training data. Its predictions are not definitive proof that a real-world attack has occurred.

---

## 🗂️ Project Structure

```text
CyberSentryFusion/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── preprocessor.joblib
│   ├── xgboost_classifier.json
│   └── threshold.json
│
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-preprocessing-training.ipynb
│   ├── 03-model-explainability.ipynb
│   └── 04-model-testing.ipynb
│
├── requirements.txt
└── README.md
```

*The structure above highlights the main application, model artifacts, and notebooks. Other repository files and directories may also be present.*

---

## 🛠️ Technology Stack

| Technology           | Purpose                                    |
| -------------------- | ------------------------------------------ |
| Python               | Core programming language                  |
| Pandas               | Data manipulation and analysis             |
| NumPy                | Numerical operations                       |
| Scikit-learn         | Preprocessing and ML utilities             |
| XGBoost              | Classification model                       |
| SHAP                 | Model explainability                       |
| Streamlit            | Interactive web application                |
| Plotly               | Interactive visualizations                 |
| Matplotlib & Seaborn | Exploratory data analysis                  |
| Joblib               | Saving and loading preprocessing artifacts |
| Jupyter Notebook     | Model development and experimentation      |
| Git & GitHub         | Version control and collaboration          |

---

## 📚 Dataset

This project uses the **UNSW-NB15 network intrusion detection dataset**, which contains network traffic records representing normal activity and different types of attacks.

The model is configured for binary classification:

| Label | Meaning        |
| ----- | -------------- |
| `0`   | Normal traffic |
| `1`   | Attack traffic |

The application predicts these two classes rather than identifying every individual attack category.

Dataset source: [UNSW-NB15 — Australian Centre for Cyber Security](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

**Dataset note:** Obtain the dataset from its official source and follow its applicable usage terms. The complete dataset does not need to be committed to GitHub to use this repository, provided you configure the local training workflow appropriately.

---

## ⚙️ Installation and Setup

### Prerequisites

* Python installed on your system.
* Git.
* A local copy of this repository.

### 1. Clone the Repository

```bash
git clone https://github.com/dudecoded/CyberSentryFusion.git
```

### 2. Navigate to the Project

```bash
cd CyberSentryFusion
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify Model Files

Ensure the following saved artifacts are present in the `models/` directory:

* `preprocessor.joblib`
* `xgboost_classifier.json`
* `threshold.json`

The dashboard depends on these artifacts to load the preprocessing pipeline, classifier, and decision threshold.

### 6. Run the Application

From the project root directory, execute:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will display a local URL in your terminal. Open that URL in your browser to access the dashboard.

---

## 📓 Exploring the Notebooks

The notebooks document the model development workflow.

| Notebook                          | Purpose                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `01-data-exploration.ipynb`       | Explore the dataset, distributions, and feature characteristics.                      |
| `02-preprocessing-training.ipynb` | Prepare the data and train the machine-learning model.                                |
| `03-model-explainability.ipynb`   | Investigate model behaviour using explainability techniques.                          |
| `04-model-testing.ipynb`          | Evaluate the trained model using classification metrics and other evaluation methods. |

Run the notebooks in sequence when reproducing the documented workflow. Ensure the dataset paths and required dependencies are configured for your environment.

---

## 📏 Evaluation

Model performance should be evaluated on held-out data using multiple metrics rather than accuracy alone.

Recommended metrics include:

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of predicted attacks that are actually attacks.
* **Recall:** Proportion of actual attacks correctly detected.
* **F1-score:** Harmonic mean of precision and recall.
* **ROC-AUC:** Ability to distinguish between the two classes across classification thresholds.
* **Confusion matrix:** Breakdown of true positives, true negatives, false positives, and false negatives.

For intrusion detection, recall is particularly important because false negatives represent attacks that the model fails to identify. However, false positives also matter because they can create unnecessary security alerts.

The decision threshold affects this trade-off. Its selection should be supported by validation results and the intended use case.

---

## ⚠️ Limitations

* CyberSentryFusion performs predictions on supplied network traffic records; it is **not a live packet-capture or real-time network monitoring system**.
* Predictions depend on the quality, format, and feature compatibility of the supplied data.
* Model performance on a benchmark dataset may not generalize to unseen real-world network environments.
* A prediction labelled `ATTACK` is a model-generated assessment, not a confirmed security incident.
* SHAP explanations describe the model's behaviour; they do not independently establish the cause of malicious activity.
* Prediction scores should not be interpreted as calibrated probabilities of a real-world attack unless probability calibration has been independently evaluated.

---

## 🔮 Future Improvements

Potential directions for extending the project include:

* Real-time network traffic ingestion.
* Integration with network monitoring tools.
* Alerting and event logging.
* Detection of individual attack categories.
* Testing on additional datasets and unseen traffic.
* Model comparison and systematic threshold optimization.
* Improved monitoring of false positives and false negatives.
* Deployment and performance optimization for larger traffic volumes.

---

## 🎯 Project Goals

CyberSentryFusion aims to demonstrate how machine learning and explainable AI can be combined to explore network intrusion detection through an accessible, interactive application.

The project emphasizes the complete ML workflow—from dataset exploration and preprocessing to model evaluation, explainability, and deployment.

---

## 👨‍💻 Author

**Uzair Siddiqui**

GitHub: [@dudecoded](https://github.com/dudecoded)

---

## 📄 License

Check the repository's `LICENSE` file for the applicable license. If the project does not yet have one, choose and add a license before presenting the project as open source.

---

*CyberSentryFusion — Explore traffic. Understand predictions. Investigate potential threats.*
