import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def get_theta() -> tuple[float, float]:
    """Get the theta values from the file .theta found in the project dir
    if not found create file and init thetas at 0

    Returns:
      tuple: Tuple containing the theta values
    """
    try:
        with open('.theta', 'r') as f:
            theta0 = float(f.readline().strip())
            theta1 = float(f.readline().strip())
            return theta0, theta1
    except FileNotFoundError:
        with open('.theta', 'w') as f:
            f.write('0.0\n0.0')
            return 0.0, 0.0


def estimate_price(theta0: float, theta1: float, milleage: float) -> float:
    """Estimate the price of a car given its milleage

    Parameters:
      milleage (int): Milleage of the car

    Returns:
      float: Estimated price of the car
    """
    return theta0 + theta1 * milleage


def main() -> int:
    """Estimate the price of a car given its milleage

    Parameters:
      ac (int): Number of arguments
      av (list): List of arguments
    """
    try:
        milleage = float(input("Enter the car milleage to estimate its price: "))
        theta0, theta1 = get_theta()
        estimated_price = estimate_price(theta0, theta1, milleage)
        print(f"Car milleage: \033[34m{milleage}\033[0m")
        print("The algorithm predicted that the car price would be...")
        print(f"\033[32m{estimated_price:0.2f}$\033[0m")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
