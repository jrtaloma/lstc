# Concrete Dense Network for Long-Sequence Time Series Clustering

This repository contains the official source code for the paper: **"Concrete Dense Network for Long-Sequence Time Series Clustering"**

---

## 📦 Requirements

- Python 3.9

It is **strongly recommended** to use a virtual environment or container.

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## 📁 Datasets

### UCR Time Series Dataset

1. Download the dataset from [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
2. Extract the contents into the `datasets` folder:

   ```
   datasets/
     └── UCRArchive_2018/
   ```

### M5 Forecasting Dataset

1. Download the dataset from the [M5 Forecasting - Accuracy competition on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

2. Extract the contents into the `datasets` folder:

   ```bash
   unzip m5-forecasting-accuracy.zip -d datasets
   rm -rf datasets/__MACOSX
   rm -f datasets/m5-forecasting-accuracy/sample_submission.csv
   ```

3. Run preprocessing script:

   ```bash
   python M5_EDA.py
   ```

4. Open and run the notebook for additional preprocessing:

   * `M5_EDA_cat_store_id.ipynb`

---

## 🚀 Training & Evaluation

Each model has its own folder containing a `run.sh` script to launch training for each dataset.

To train a model:

```bash
./run.sh
```

You can customize hyperparameters and set random seeds within the script.

To compute average performance metrics (across different seeds):

```bash
python results.py
```
