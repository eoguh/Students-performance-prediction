import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler




# Load your data from a CSV file (replace 'data.csv' with your actual file)
students_data = pd.read_csv('Students_data.csv')



# Split the data into features (X) and target (y)
X = students_data.drop(columns=['GRADE'])
y = students_data['GRADE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# Modelling

# Choose your machine learning model (replace with your chosen model)
from sklearn.linear_model import LinearRegression

# Define and train the model
model = LinearRegression()
model.fit(X_train, y_train)






# Prediction

# Evaluate model performance (replace with your chosen metric)
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Make a prediction for a new student (replace with new student data)
new_student_data = [...]  # Replace with actual data for new student
new_prediction = model.predict([new_student_data])
print(f"Predicted performance: {new_prediction[0]}")



"""
# Recommendations

# Analyze feature importances (if using a model that supports it)
importances = model.feature_importances_  # Replace with relevant attribute for your model
feature_names = features.columns

# Identify features with high importance
important_features = feature_names[importances.argsort()[-3:]]  # Top 3 most important features

# Based on the model's prediction and important features, formulate recommendations
if new_prediction < threshold:  # Replace 'threshold' with your performance benchmark
    recommendations = [f"Consider increasing your daily study time by X hours based on the model's analysis."
                       f" Students with similar profiles who participated in {important_features[0]} showed improved performance."]
else:
    recommendations = ["Your predicted performance is good, keep up the good work!"]

print(f"Recommendations for improvement: {recommendations}")

"""