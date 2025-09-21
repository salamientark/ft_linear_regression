import os as os
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_data(price: list, km: list):
    """Plot the data using matplotlib

    Parameters:
      price (list): List of prices
      km (list): List of kilometers
    """
    plt.xlabel('km')
    plt.ylabel('price')
    plt.plot(km, price, 'o', color='blue')
    plt.show()


def main():
    try:
        data = load_data()
        km = data["km"].tolist()
        price = data["price"].tolist()
        print(km, type(km))
        print(price, type(price))
        plot_data(price, km)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
