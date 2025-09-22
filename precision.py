import numpy as np
from typing import Callable
from train import load_data
from estimate_price import get_theta, estimate_price


def mean_square_error(
        input: np.ndarray,
        output: np.ndarray,
        theta0: float,
        theta1: float,
        hypothesis: Callable[[float, float, float], float]
        ) -> float:
    """Calculate the mean square error between the output and the hypothesis

    Parameters:
      input (np.ndarray): Input values
      output (np.ndarray): Output values
      theta0 (float): Theta0 value
      theta1 (float): Theta1 value
      hypothesis (Callable[[float, float, float], float]): Hypothesis function

    Returns:
      float: Mean square error
    """
    data_size = len(output)
    sum = 0
    for i in range(data_size):
        sum += (output[i] - hypothesis(theta0, theta1, input[i])) ** 2
    mse = (1 / data_size) * sum
    return mse


def mean_absolute_error(
        input: np.ndarray,
        output: np.ndarray,
        theta0: float,
        theta1: float,
        hypothesis: Callable[[float, float, float], float]
        ) -> float:
    """Calculate the mean absolute error between the output and the hypothesis

    Parameters:
      input (np.ndarray): Input values
      output (np.ndarray): Output values
      theta0 (float): Theta0 value
      theta1 (float): Theta1 value
      hypothesis (Callable[[float, float, float], float]): Hypothesis function

    Returns:
      float: Mean absolute error
    """
    data_size = len(output)
    sum = 0
    for i in range(data_size):
        sum += np.absolute(output[i] - hypothesis(theta0, theta1, input[i]))
    mae = (1 / data_size) * sum
    return mae


def variance(
        input: np.ndarray,
        output: np.ndarray,
        theta0: float,
        theta1: float,
        hypothesis: Callable[[float, float, float], float]
        ) -> float:
    """Calculate the variance calculated by the model

    Parameters:
      input (np.ndarray): Input values
      output (np.ndarray): Output values
      theta0 (float): Theta0 value
      theta1 (float): Theta1 value
      hypothesis (Callable[[float, float, float], float]): Hypothesis function

    Returns:
      float: variance explained by model
    """
    data_size = len(output)
    output_mean = np.mean(output)
    sum_numerator = sum_denominator = 0
    for i in range(data_size):
        sum_numerator += (output[i]
                          - hypothesis(theta0, theta1, input[i])) ** 2
        sum_denominator += (output[i] - output_mean) ** 2
    variance = 1 - (sum_numerator / sum_denominator)
    return variance


def print_result(
        theta0: float,
        theta1: float,
        mse: float,
        rmse: float,
        mae: float,
        var: float
        ) -> None:
    """Measure precision/error of the model using different algorithm
    (MSE, RMSE, MAE, VAR)

    Parameters:
      theta0 (float): Theta0 value
      theta1 (float): Theta1 value
      mse (float): Mean square error
      rmse (float): Root mean square error
      mae (float): Mean absolute error
      var (float): Variance
    """
    # ANSI color codes
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

    print(f"{HEADER}{BOLD}Model Parameters:{ENDC}")
    print(f"{OKBLUE}  Theta0: {theta0:.4f}{ENDC}")
    print(f"{OKBLUE}  Theta1: {theta1:.4f}{ENDC}")
    print(f"{OKCYAN}{BOLD}Mean Square Error (MSE):{ENDC} {mse:.4f}")
    print(f"{OKCYAN}{BOLD}Root Mean Square Error (RMSE):{ENDC} {rmse:.4f}")
    print(f"{OKCYAN}{BOLD}Mean Absolute Error (MAE):{ENDC} {mae:.4f}")
    print(f"{OKGREEN}{BOLD}Variance (VAR):{ENDC} {var:.4f}")


def main():
    """Show model precision/error"""
    try:
        data = load_data()
        km = data["km"].to_numpy()  # Not normalized data
        price = data["price"].to_numpy()
        theta0, theta1 = get_theta()  # Not scaled theta
        mse = mean_square_error(km, price, theta0, theta1, estimate_price)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(km, price, theta0, theta1, estimate_price)
        var = variance(km, price, theta0, theta1, estimate_price)
        print_result(theta0, theta1, mse, rmse, mae, var)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
