import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

# START CODE EXECUTION TIMER
startTime = datetime.now()
print("\nStarted script at: {}".format(startTime))
print("\n* * * * BEGIN CODE EXECUTION * * * *\n")


#############################


# Load the data from the CSV file
data = pd.read_csv('data/T_10Y_weekly_6_12_2023.csv')

# Split the data into training and testing sets
train_data = data[:int(0.8*len(data))] # 80% training
test_data = data[int(0.8*len(data)):] # 20% testing

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(train_data[['Open', 'High', 'Low', 'Volume']], train_data['Close'])

# Predict the stock price on the test data
predictions = model.predict(test_data[['Open', 'High', 'Low', 'Volume']])

#############################


# END CODE EXECUTION TIMER
print("\n* * * *  END  CODE EXECUTION * * * *\n")
endTime = datetime.now()
print("Ended script at: {}".format(endTime))
print("Script execution time: {}\n".format(endTime - startTime))

# Plot the predicted and actual stock prices
plt.plot(test_data['Date'], predictions, label='Predicted (Linear Regression)', color='red')
plt.plot(test_data['Date'], test_data['Close'], label='Actual (Fidelity.com)', color='green')
plt.xlabel('Weeks since 2013-06-17')
plt.xscale('linear')
plt.ylabel('Stock Price (USD $)')
plt.title('\'T\' Stock Price Prediction - June 2013 to June 2023')
plt.legend()
plt.show()