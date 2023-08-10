# Rossmann Store Sales

Rossman Kaggle Mini-Competition: Forecast sales using store, promotion, and competitor data

This is a Kaggle competition and can be found [here](https://www.kaggle.com/competitions/rossmann-store-sales/overview)

## Language Used : Python --version: 3.10.6

## Setup <---------- TO EDIT

1 - Create a conda environment

2 - Install Pip
```conda install pip```

3 - ```pip install -r requirements.txt```

4 - Run the `ADD_file_name` Notebook to make predictions


## Feature <---------- TO EDIT

In the model we selected the following features:

- `Promo`: Mean encoding (including `Store`) [new var: `PromoStoreMean`]
- `DayofWeek`: Mean encoding
- `Date`: extract `Month` and then Mean encoding [new var: `MonthMean`]
- `Storetype`: Mean encoding
- `StateHoliday`: Mean encoding
- `SchoolHoliday`: Mean encoding
- `Promo2`: Mean encoding
