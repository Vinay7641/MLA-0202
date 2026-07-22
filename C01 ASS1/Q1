import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv(r"C:\Users\shaik\Downloads\Housing.csv")


print("First 5 Rows of Dataset:")
print(data.head())


print("\nMissing Values:")
print(data.isnull().sum())


encoder = LabelEncoder()

categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])


X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = r2_score(y_test, y_pred)

print("\nPredicted House Prices:")
print(y_pred)

print("\nActual House Prices:")
print(y_test.values)

print("\nR2 (Accuracy) Score:", accuracy)
