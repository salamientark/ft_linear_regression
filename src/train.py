import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from estimate_price import estimate_price, get_theta


### PROGRAM CONSTANT
ALPHA = 0.01    # Learning rate
ITERATIONS = 100   # Number of iterations


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


def plot_data(price: np.ndarray, km: np.ndarray, old_theta0: float,
        old_theta1: float, new_theta0: float, new_theta1: float):
    """Plot the data using matplotlib

    Parameters:
      price (np.ndarray): List of prices
      km (np.ndarray): List of kilometers
    """
    plt.xlabel('km')
    plt.ylabel('price')
    plt.plot(km, price, 'o', color='blue', label='data')
    plt.plot(km, old_theta0 + old_theta1 * km, color='orange',
             label='old hypothesis')
    plt.plot(km, new_theta0 + new_theta1 * km, color='green',
            label='new hypothesis')
    plt.legend()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def get_new_param_value(
        features: np.ndarray,
        output: np.ndarray,
        theta0: float,
        theta1: float
    ) -> tuple[float, float]:
    """Calculate new parameter value using gradient descent algorithm

    Parameters:
      features/ (np.ndarray): List of features (1, milleage)
      output (np.ndarray): List of outputs (price)
      theta0 (float): Current theta0 value
      theta1 (float): Current theta1 value

    Returns:
      float, float: New parameter value (theta0 and theta1)
    """
    data_size = len(output)
    sum0, sum1 = 0, 0
    for i in range(data_size):
        estimated_price = estimate_price(theta0, theta1, features[i])
        sum0 += estimated_price - output[i]
        sum1 += (estimated_price - output[i]) * features[i]
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


def raw_to_scaled(
        theta0: float,
        theta1: float,
        mean : float,
        std: float
    ) -> tuple[float, float]:
    """Convert raw theta values to scaled theta values

    Parameters:
      theta0 (float): Raw theta0 value
      theta1 (float): Raw theta1 value
      mean (float): Mean of the features
      std (float): Standard deviation of the features

    Returns:
      float, float: Scaled theta0 and theta1 values
    """
    scaled_theta0 = theta0 + theta1 * mean
    scaled_theta1 = theta1 * std
    return scaled_theta0, scaled_theta1


def scaled_to_raw(
        theta0: float,
        theta1: float,
        mean : float,
        std: float
    ) -> tuple[float, float]:
    """Convert scaled theta values to raw theta values

    Parameters:
      theta0 (float): Scaled theta0 value
      theta1 (float): Scaled theta1 value
      mean (float): Mean of the features
      std (float): Standard deviation of the features

    Returns:
    float, float: Raw theta0 and theta1 values
    """
    raw_theta0 = theta0 - theta1 * mean / std
    raw_theta1 = theta1 / std
    return raw_theta0, raw_theta1


def print_result(
        old_theta0: float,
        old_theta1: float,
        new_theta0: float,
        new_theta1: float
        ) -> None:
    """Explain what the train did on the model

    Parameters:
        old_theta0 (float): Old theta0 value
        old_theta1 (float): Old theta1 value
        new_theta0 (float): New theta0 value
        new_theta1 (float): New theta1 value
    """
    print(f"Training Result after {ITERATIONS} iteration(s):")
    print(f"  theta0: {old_theta0:.6f} -> {new_theta0:.6f}")
    print(f"  theta1: {old_theta1:.6f} -> {new_theta1:.6f}")


def main():
    """Train the model using the data from data.csv"""
    try:
        data = load_data()
        tmp_km = data["km"].to_numpy() # Not normalized data
        mean = tmp_km.mean()
        std = tmp_km.std()
        km = np.array([(x - mean) / std for x in tmp_km]) # normalized data
        price = data["price"].to_numpy()
        raw_theta0, raw_theta1 = get_theta() # Not scaled theta
        scaled_theta0, scaled_theta1 = raw_to_scaled(
            raw_theta0, raw_theta1, mean, std) # Scaled theta

        for i in range(ITERATIONS) :
            scaled_theta0, scaled_theta1 = get_new_param_value(km, price, scaled_theta0, scaled_theta1)
        new_theta0, new_theta1 = scaled_to_raw(scaled_theta0, scaled_theta1, mean, std)
        update_theta(new_theta0, new_theta1)
        print_result(raw_theta0, raw_theta1, new_theta0, new_theta1)
        plot_data(price, tmp_km, raw_theta0, raw_theta1, new_theta0, new_theta1)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
