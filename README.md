# Churn_prediction

# 📝 Description

An important financial institution is interested in analyzing its client database to increase revenue generated from credit cardholders while reducing churn rates. With a churn rate exceeding 15% and showing an upward trend, the CEO has tasked the marketing team with initiating a campaign for client retention.

The dataset, named `BankChurners.csv`, was provided by the client. During preprocessing (refer to **Preprocessing Data**), three columns were dropped per client advice, including the last two columns and the client number column, as they were deemed irrelevant for data processing. To ensure data cleanliness, 'Unknown' values were handled differently based on the modeling approach.

Label encoding was employed to transform string values into integers, a necessary step for model training.

# 🔧 Installation

To utilize the model with the existing data, no installation is required. Simply access StreamLite via: [StreamLite Link].

For implementation with custom data, clone the repository and follow the instructions detailed in the `requirements.txt` file.

# 📂 Repo Structure

- venv
- data
  - BankChurners.csv
  - exploration_data.ipynb
- model1
  - prediction.py
  - train.py
- README.md

# 🚀 Usage

Downloading the model facilitates the bank in predicting whether a customer will churn within a specified time frame. When using your data, ensure it follows the same template as the provided CSV file.

# ⚠️ **Data Sources**

The data utilized in this project is sourced from Kaggle's [Credit Card Customers](https://www.kaggle.com/sakshigoyal7/credit-card-customers).

# 🖼️ Visuals

(Include any visuals here)

# 👥 Contributors

Despite being a solo project, I extend my gratitude to my colleagues at Becodian. Their invaluable brainstorming sessions and assistance in debugging the entire script were instrumental in completing this project.

# ⏰ Timeline

This project was completed within a timeframe of 5 working days.

# 📌 Personal Note

(Add any personal notes or acknowledgments here)
