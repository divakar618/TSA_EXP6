# Ex.No: 6               HOLT WINTERS METHOD
### Date: 
## Name : Divakar R
## Register number : 212222240026


### AIM:
To implement Holt-Winters model on Electric Production Data Set and make future predictions

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```python
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('powerconsumption.csv')

# Preprocess the data
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# Display the first five rows
print("First Five Rows:")
print(df.head())

# Split the data into training and test sets
train_size = int(len(df) * 0.8)  # 80% training data
train, test = df[:train_size], df[train_size:]

# Fit the Holt-Winters model
holt_winters_model = ExponentialSmoothing(train, 
                                           trend='add', 
                                           seasonal='add', 
                                           seasonal_periods=12)  # Adjust seasonal_periods based on your data
holt_winters_fit = holt_winters_model.fit()

# Make predictions
test_predictions = holt_winters_fit.forecast(len(test))
final_predictions = holt_winters_fit.predict(start=test.index[0], end=df.index[-1])

# Calculate and print the Mean Squared Error for test predictions
mse = mean_squared_error(test, test_predictions)
print(f'Mean Squared Error of Test Predictions: {mse}')

# Visualize Test Predictions
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data', color='blue')
plt.plot(test, label='Test Data', color='orange')
plt.plot(test_predictions, label='Test Predictions', color='green', linestyle='--')
plt.title('Holt-Winters Test Predictions')
plt.xlabel('Date')
plt.ylabel('Electric Production')
plt.legend()
plt.grid()
plt.show()

# Visualize Final Predictions
plt.figure(figsize=(12, 6))
plt.plot(df, label='Actual Data', color='blue')
plt.plot(final_predictions, label='Final Predictions', color='red', linestyle='--')
plt.title('Holt-Winters Final Predictions')
plt.xlabel('Date')
plt.ylabel('Electric Production')
plt.legend()
plt.grid()
plt.show()

```


### OUTPUT:


TEST_PREDICTION

<img width="1007" alt="376884614-d56c85ba-0bb4-4399-8fea-b7c0c9da3a86" src="https://github.com/user-attachments/assets/0fcbe2f2-2921-4269-b6f8-afe82edd1f2b">



FINAL_PREDICTION

<img width="861" alt="376884664-ee9e01b3-4430-40c4-9570-eafb26adafbd" src="https://github.com/user-attachments/assets/6efc6545-6123-47a2-b193-9c14fae6085e">



### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
