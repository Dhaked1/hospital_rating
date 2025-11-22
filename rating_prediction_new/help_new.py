import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle

# Load the dataset
data = pd.read_csv('help_new.csv')

# Define input and output columns
input_columns = [
    'H_BATH_HELP_A_P', 'H_BATH_HELP_SN_P', 'H_BATH_HELP_U_P',
    'H_CALL_BUTTON_A_P', 'H_CALL_BUTTON_SN_P', 'H_CALL_BUTTON_U_P',
    'H_COMP_3_A_P', 'H_COMP_3_SN_P', 'H_COMP_3_U_P'
]
output_column = 'H_COMP_3_STAR_RATING'

# Select relevant columns
data = data[input_columns + [output_column]]

# Handle missing values
data.replace(['Not Available', '', 'NA', 'N/A'], np.nan, inplace=True)

# Remove rows with any NaN values in the selected columns
data = data.dropna()

# Ensure all input columns are numeric
for col in input_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Ensure the output column is numeric
data[output_column] = pd.to_numeric(data[output_column], errors='coerce')

# Drop any rows that still have NaN after conversion
data = data.dropna()

# Extract features (X) and target (y)
X = data[input_columns].values
y = data[output_column].values

# Convert star ratings to 0-based (0 to 4 for 1 to 5 stars) and one-hot encode
y = to_categorical(y - 1)  # Subtract 1 to make classes 0-based (0 to 4)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
with open('help_new_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')  # 5 output classes for star ratings 1-5
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model in Keras format
model.save('help_new.keras')

# Save the model in pickle format
with open('help_new.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and scaler saved successfully.")