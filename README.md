# Disaster Response

This project aims to help disaster response agencies easily filter urgent messages (from social media or any other text source) that need immediate attention.

The project includes a machine learning pipeline that trains on a given dataset to make predictions on new messages, as well as a web interface which allows emergency workers to input a new message and get classification results in several categories.

## Run Locally

Clone the project

```bash
git clone https://github.com/cameron-dryden/disaster-response
```

Go to the project directory

```bash
cd disaster-response
```

Start the web server

```bash
cd app
python run.py
```

## Model Setup

To run ETL pipeline that cleans data and stores in database

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

To run ML pipeline that trains classifier and saves

```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

## File Structure Overview

#### Data Folder

```bash
./data/
```

| File                      | Description                                                                                |
| :------------------------ | :----------------------------------------------------------------------------------------- |
| `disaster_categories.csv` | Contains all the raw CSV data for the categories that are realted to a message.            |
| `disaster_messages.csv`   | Contains all the raw CSV data for the messages.                                            |
| `DisasterResponse.db`     | SQLite Database which contains the cleaned data. _This is the result of the script below._ |
| `process_data.py`         | Contains the code for the ETL pipeline that cleans data and stores it in a database        |

#### Models Folder

```bash
./models/
```

| File                  | Description                                                                           |
| :-------------------- | :------------------------------------------------------------------------------------ |
| `classifier.pkl`      | Contains a pre-trained model that is used by the web interface to perform predictions |
| `train_classifier.py` | Contains the code for the ML pipeline that trains classifier and saves it             |

#### App Folder

```bash
./app/
```

| File     | Description                                                                                        |
| :------- | :------------------------------------------------------------------------------------------------- |
| `run.py` | Starts a web server which hosts the web interface to interact with the model and perform inference |

## Model Overview and Performance

![ML Pipeline Diagram](ML_Pipeline.png?raw=true)

**The final model scored:** \
f1 score: 94.59%, precision: 94.52%, recall: 95.22%

## Acknowledgements

- [Dataset provided by Appen](https://appen.com/)
- [Scikit-Learn Machine Learning](https://scikit-learn.org/stable/)
- [Pandas Data Analysis Tool](https://pandas.pydata.org/docs/)
