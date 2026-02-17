import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing

DATA_FILE = "prices.csv"

def prepare_data(df, forecast_col="close", forecast_out=5):
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    label.dropna(inplace=True)
    y = np.array(label)
    return X, y, X_lately

def predict_stock(symbol, forecast_out):
    df = pd.read_csv(DATA_FILE)
    df = df[df["symbol"] == symbol].copy()

    if df.empty:
        print("Symbol not found in dataset.")
        return

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    X, y, X_lately = prepare_data(df, forecast_out=forecast_out)

    model = pickle.load(open("stock_model.pkl", "rb"))
    forecast = model.predict(X_lately)

    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_out
    )

    print(f"\nNext {forecast_out} day predictions for {symbol}:")
    for date, price in zip(future_dates, forecast):
        print(f"{date.date()}  →  {price:.2f}")

    train_data = df[:-forecast_out]

    plt.figure(figsize=(12,6))
    plt.plot(train_data.index, train_data["close"], label="Historical Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{symbol} Historical Prices")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(future_dates, forecast, marker='o', linestyle='--', label="Forecast")
    plt.xlabel("Future Date")
    plt.ylabel("Predicted Price")
    plt.title(f"Next {forecast_out} Days Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., GOOG): ").upper()
    days = int(input("Enter number of days to predict: "))

    if days <= 0:
        print("Please enter a positive number.")
    else:
        predict_stock(symbol, days)