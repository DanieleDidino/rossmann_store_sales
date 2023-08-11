# Rossmann Store Sales

Rossman Kaggle Mini-Competition: Forecast sales using store, promotion, and competitor data

This is a Kaggle competition and can be found [here](https://www.kaggle.com/competitions/rossmann-store-sales/overview)

### Language Used : Python --version: 3.10.6

## Setup <---------- TO EDIT

1. Create a virtual environment: `python3 -m venv .venv`
2. Activate virtual environment: `source .venv/bin/activate`
3. Install the packages: `pip install -r requirements.txt`
4. Run the jupyter Notebooks (descriptives, model_development etc.)
5. To make predicitons run `ADD_FILE_NAME` in terminal or use this [Streamlit app](ADD_https)

## Repository Contents

1. data: folder containing data.
2. 1_EDA: 
3. 2_baseline_model:
4. functions.py : python scripts containing all functions used.
5. readme.md 
6. requirements.txt
7. solution.ipynb 
8. pipeline.py : python script to run holdout data.
9. pipeline : Folder containing pipeline.

## Feature <---------- TO EDIT

In the model we selected the following features:

- `Promo`: Mean encoding (including `Store`) [new var: `PromoStoreMean`]
- `DayofWeek`: Mean encoding
- `Date`: extract `Month` and then Mean encoding [new var: `MonthMean`]
- `Storetype`: Mean encoding
- `StateHoliday`: Mean encoding
- `SchoolHoliday`: Mean encoding
- `Promo2`: Mean encoding
