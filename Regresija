import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


df = pd.read_csv("delivery_data.csv")  

# Replace missing or empty values
df = df.replace(r'^\s*$', np.nan, regex=True)


categorical_cols = ["Time_of_Day", "Weather", "Traffic_Level", "Vehicle_Type"]
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


df = df.fillna(df.mean(numeric_only=True))


X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"].values.reshape(-1,1)


scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)


val_size = 0.2 / 0.7
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=42
)


model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["mean_absolute_error"])


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=1,
    min_delta=0.01,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)


train_loss_norm = np.array(history.history["loss"]) / history.history["loss"][0]
val_loss_norm = np.array(history.history["val_loss"]) / history.history["val_loss"][0]

plt.figure(figsize=(8,5))
plt.plot(train_loss_norm, label="Training Loss")
plt.plot(val_loss_norm, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()


y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    mape = (100 / n) * np.sum(np.abs((y_pred - y_true) / y_true))
    return mape

train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

print(f"Training MAPE: {train_mape:.2f}%")
print(f"Validation MAPE: {val_mape:.2f}%")
print(f"Testing MAPE: {test_mape:.2f}%")
