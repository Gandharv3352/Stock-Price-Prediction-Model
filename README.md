<div align="center">
<h1 align="center">📈 Stock Price Prediction Model</h1>

<p align="center">
  Forecast Market Trends with Machine Learning
  <br />
  <br />
  <a href="https://github.com/Gandharv3352/Stock-Price-Prediction-Model"><strong>Explore the Repository »</strong></a>
  <br />
  <br />
  <a href="#downloads">Download Files</a>
  ·
  <a href="#getting-started">Run Locally</a>
  ·
  <a href="https://github.com/Gandharv3352/Stock-Price-Prediction-Model/issues">Report Bug</a>
</p>
</div>

---

<!-- TABLE OF CONTENTS -->

<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#why-this-project">Why This Project?</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#languages-used">Languages Used</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#downloads">Downloads</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#model-details">Model Details</a></li>
    <li><a href="#popular-stock-symbols">Popular Stock Symbols</a></li>
    <li><a href="#future-work">Future Work</a></li>
  </ol>
</details>

## ⚠️ Disclaimer

This project is for educational purposes only and not financial advice.

---

## About The Project

This project demonstrates a **complete machine learning workflow** for forecasting stock prices using historical data.

It highlights **production-ready practices**, including:

* Data preprocessing & scaling
* Model training & serialization
* Forecast visualization
* Command-line based predictions

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Why This Project?

The **Stock Price Prediction Model** shows how machine learning can assist in financial trend analysis.

### Key Features

* 📊 **Historical Trend Visualization**
  Displays past stock performance.

* 🔮 **Future Price Forecasting**
  Predict upcoming stock prices for user-selected days.

* ⚙️ **Reusable Prediction Tool**
  Users can input any supported stock symbol.

* 📦 **Portable Model**
  Load the trained model instantly using Pickle.

* 📈 **Clear Forecast Visualizations**
  Separate historical and future trend graphs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### 🧠 Built With

This project was built using the following major libraries and tools:

* ![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python\&logoColor=white)
* ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-17A2B8?style=for-the-badge\&logo=pandas\&logoColor=white)
* ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
* ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-DD0031?style=for-the-badge)
* ![Pickle](https://img.shields.io/badge/Pickle-Serialization-6C757D?style=for-the-badge\&logo=python\&logoColor=white)
* ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge\&logo=jupyter\&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Languages Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-100%25-blue?style=for-the-badge&logo=python" />
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Project Structure

```plaintext
Stock-Price-Prediction-Model/
│
├── Stock_Prediction_Code.py   # Prediction script
├── Stock_Prediction.ipynb     # Training & experimentation
├── stock_model.pkl            # Trained regression model
├── prices.csv                 # Historical stock dataset
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Downloads

- 📓 **Jupyter Notebook (EDA + Model Training)**  
  👉 [Download Titanic_Model.ipynb](https://github.com/Gandharv3352/Stock-Price-Prediction-Model/blob/main/Stock_Prediction.ipynb)

- 🧠 **Trained Model (`.pkl`)**  
  👉 [Download titanic_pipeline.pkl](https://github.com/Gandharv3352/Stock-Price-Prediction-Model/blob/main/stock_model.pkl)

> ⚠️ Ensure the same library versions are used when loading the pickle file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

* Python 3.8+
* pip

### Installation

1️⃣ Clone the repository

```bash
git clone https://github.com/Gandharv3352/Stock-Price-Prediction-Model.git
cd Stock-Price-Prediction-Model
```

2️⃣ Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

3️⃣ Run the prediction tool

```bash
python Stock_Prediction_Code.py
```

4️⃣ Enter inputs

```
Enter stock symbol: GOOG
Enter number of days to predict: 10
```

---

## Model Details

* Algorithm: **Linear Regression**
* Feature Used: Closing Price
* Forecast Horizon: User-defined
* Scaling: Standardization
* Output: Future price predictions & trend visualization

---

## Popular Stock Symbols

| Company   | Symbol |
| --------- | ------ |
| Apple     | AAPL   |
| Microsoft | MSFT   |
| Google    | GOOG   |
| Amazon    | AMZN   |
| Tesla     | TSLA   |
| Meta      | META   |
| Netflix   | NFLX   |
| NVIDIA    | NVDA   |
| Intel     | INTC   |
| AMD       | AMD    |

---

⭐ If you found this project helpful, consider giving it a star!

---
