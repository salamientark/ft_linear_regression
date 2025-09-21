import os as os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from estimate_price import estimate_price, get_theta


# Learning rate
ALPHA = 0.001


def load_data() -> pd.DataFrame:
    """Load data from the file data.csv if found in the project dir

    Return:
        pd.DataFrame: DataFrame containing the data from data.csv
    """
    for root, dirs, files in os.walk('.'):
        if 'data.csv' in files:
            return pd.read_csv('data.csv')
        if 'src' in dirs:
            os.chdir('src')
        else:
            os.chdir('..')
        for root, dirs, files in os.walk('.'):
            if 'data.csv' in files:
                return pd.read_csv('data.csv')
            raise FileNotFoundError("data.csv not in project directory")


def plot_data(price: list, km: list, new_theta0: float, new_theta1: float):
    """Plot the data using matplotlib

    Parameters:
      price (list): List of prices
      km (list): List of kilometers
    """
    theta0, theta1 = get_theta()
    print(theta0, theta1)
    print(new_theta0, new_theta1)
    print("Price: ", price)
    km = np.array(km)
    plt.xlabel('km')
    plt.ylabel('price')
    plt.plot(km, price, 'o', color='blue')
    plt.plot(km, theta0 + theta1 * km, color='orange')  # Plot old theta
    plt.plot(km, new_theta0 + new_theta1 * km, color='green')  # Plot new theta
    plt.show()


def get_new_param_value(features: list, output: list) -> tuple[float, float]:
    """Calculate new parameter value using gradient descent algorithm

    Parameters:
      features/ (list): List of features (1, milleage)
      output (list): List of outputs (price)

    Returrns:
      float, float: New parameter value (theta0 and theta1)
    """
    data_size = len(output)
    theta0, theta1 = get_theta()
    sum0, sum1 = 0, 0
    for i in range(data_size):
        print(features[i], output[i], estimate_price(theta0, theta1, features[i]))
        sum0 += estimate_price(theta0, theta1, features[i]) - output[i]
        sum1 += (estimate_price(theta0, theta1, features[i]) - output[i]) * features[i]
    new_theta0 = theta0 - ALPHA * (1 / data_size) * sum0
    new_theta1 = theta1 - ALPHA * (1 / data_size) * sum1
    return new_theta0, new_theta1


def update_theta(theta0: float, theta1: float):
    """Update the theta values in the file .theta

    Parameters:
      theta (tuple): Tuple containing the theta values
    """
    with open('.theta', 'w') as f:
        f.write(f"{theta0}\n{theta1}")


def main():
    """Train the model using the data from data.csv"""
    try:
        data = load_data()
        km = data["km"].tolist()
        price = data["price"].tolist()
        new_theta0, new_theta1 = get_theta()
        for i in range(100) :
            new_theta0, new_theta1 = get_new_param_value(km, price)
            update_theta(new_theta0, new_theta1)
        plot_data(price, km, new_theta0, new_theta1)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
