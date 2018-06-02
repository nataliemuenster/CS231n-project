# CS231n-project

## Setup
1. Run `pip install -r requirements.txt`
2. Create a `data` directory and add in `train.csv` and `test.csv` from the [Google Landmarks Kaggle Page](https://www.kaggle.com/google/google-landmarks-dataset/data).
3. Run `split-csv.py` to split the large data CSV files and make the downloading process more manageable.
4. Run `hydrate-data.py` to download all the images from a given CSV file.