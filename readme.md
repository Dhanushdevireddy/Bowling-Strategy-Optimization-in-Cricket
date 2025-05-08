## Folder Structure

```
.
├── Experiments
│   ├── abliation
│   │   ├── mlp.ipynb                  # Ablation study using the MLP model.
│   │   ├── randomforest.ipynb         # Ablation experiment with Random Forest.
│   │   └── xgboost.ipynb              # Ablation analysis using XGBoost.
│   ├── Qualitative_Analysis_test.ipynb  # Qualitative evaluation of model predictions.
│   └── Random_Forest.ipynb           # Experiment notebook focused on Random Forest performance.
├── MLP
│   ├── infer_mlp.py                  # Runs inference using the trained MLP model.
│   └── train.ipynb                   # Trains the MLP model on the dataset.
├── Preprocessed_Data.csv             # Preprocessed dataset ready for modeling.
├── Preprocessing_and_EDA.ipynb      # Performs data cleaning and exploratory data analysis.
├── Random_Forest
│   ├── infer_rf.py                   # Inference script for the trained Random Forest model.
│   └── train.ipynb                   # Notebook for training the Random Forest model.
├── XGBoost
│   ├── infer_xgb.py                  # Inference script for the trained XGBoost model.
│   └── train.ipynb                   # Notebook to train the XGBoost model.
├── readme.md                         # Project documentation and usage instructions.
├── requirements.txt                  # List of dependencies to install for running the project.
```

## Dataset

Download the dataset used in the project from the following GitHub repository:

🔗 [https://github.com/ritesh-ojha/T20-World-Cup-Dataset](https://github.com/ritesh-ojha/T20-World-Cup-Dataset)

Save the required CSV files in the root directory of the project.

## Pre-trained Models

Download the pre-trained models (Random Forest, XGBoost, and MLP) from this Google Drive link:

🔗 [https://drive.google.com/file/d/1mg0Uq2rzXsjO6OxLMH0p9rpApDtdTJgw/view?usp=sharing](https://drive.google.com/file/d/1mg0Uq2rzXsjO6OxLMH0p9rpApDtdTJgw/view?usp=sharing)

Place the downloaded models in the respective model folders (MLP, Random\_Forest, XGBoost) as required by their inference scripts.

## Installation

To install all necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Project

Launch the notebooks to explore training, evaluation, and ablation studies:

```bash
jupyter notebook
```

There is no eval.py script as there was no benchmark model used in this project. Instead, evaluation is handled within the respective notebooks.

## Models Implemented

* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)

Each model comes with training and inference scripts, and has been evaluated qualitatively and through ablation experiments.

---