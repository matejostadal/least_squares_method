import numpy as np
import matplotlib.pyplot as plt


def least_squares(points, values, weights, poly_degree=1, phi_polynoms=None):
    """Performs the least squares method for given input."""

    test_input(points, values, weights, poly_degree, phi_polynoms)

    if not phi_polynoms:
        phi_functions = construct_power_functions(poly_degree + 1)
        phi_polynoms = construct_phi_polynoms(poly_degree + 1)
    else:
        phi_functions = construct_custom_functions(phi_polynoms)

    # calculates the values of the left side of the system
    equations_system_left = np.empty((poly_degree + 1, poly_degree + 1), float)

    for j in range(poly_degree + 1):
        for i in range(j, poly_degree + 1):
            equations_system_left[i][j] = equations_system_left[j][i] = discrete_dot_product(
                phi_functions[j],
                phi_functions[i],
                points,
                weights,
            )

    # calculates the values of the right side of the system
    equations_system_right = np.empty(poly_degree + 1, float)

    for k in range(poly_degree + 1):
        equations_system_right[k] = discrete_dot_product(
            # lambda(points[0]) => values[0]  (lambda represents the f function)
            lambda point: values[points.index(point)],
            phi_functions[k],
            points,
            weights
        )

    # print(equations_system_left)
    # print(equations_system_right)

    # numpy to solve the system of linear equations
    # coefficients = [c0, c1, ..., cm]
    coefficients = np.linalg.solve(equations_system_left, equations_system_right)

    # finds the approximation polynom
    # counts the sum of ci * phii(x)
    approx_polynom = []

    for coeff, phi in zip(coefficients, phi_polynoms):
        approx_polynom = np.polyadd(approx_polynom, np.polymul(coeff, phi))

    # returns a list of coefficients of approximation polynom (sorted decreasingly by "importance")
    return approx_polynom


def test_input(points, values, weights, poly_degree, phi_polynoms):
    """Checks if all input vectors are of the same dimension.
    Then checks if wanted polynom degree can be found with given phi_functions.

    Raises:
        ValueError: if dimensions of input vectors don't match or the count of phi functions is not correct.

    """
    # vectors dimension check
    if not same_length([points, values, weights]):
        raise ValueError("Method could not be performed. Check the dimensions of given input vectors.")

    # degree suitability check
    if phi_polynoms:
        if poly_degree != len(phi_polynoms) - 1:
            raise ValueError(
                "Method could not be performed. Incorrect count of phi functions or degree of wanted approximation polynom."
                )

        for phi_polynom in phi_polynoms:
            if len(phi_polynom) - 1 > poly_degree:
                raise ValueError(
                    "Method could not be performed. Phi function of larger degree than wanted approximation polynom."
                    )

    # checking for reoccuring points
    if any([points.count(point) > 1 for point in points]):
        raise ValueError("Method could not be performed. Reoccuring points.")


def same_length(values):
    """Check if each value in values is of the same length."""
    return len(set([len(value) for value in values])) <= 1


def discrete_dot_product(func1, func2, points, weights):
    """Calculates the discrete dot product for given functions, points and weights."""
    return sum(
        [weight * func1(point) * func2(point) for point, weight in zip(points, weights)]
    )


def construct_power_functions(max_power):
    """Constructs a list of power functions sorted by exponent from 0 to max_power."""
    power_functions = []

    for exponent in range(max_power):
        power_functions.append(create_power_function(exponent))

    return power_functions


def create_power_function(exponent):
    """Creates a power function with given exponent."""
    return lambda point: point ** exponent


def construct_phi_polynoms(max_power):
    """Constructs a polynomial representation of a function in a form: x^power."""
    phi_polynoms = []

    for power in range(max_power):
        polynom = [0] * (power + 1)
        polynom[0] = 1
        phi_polynoms.append(polynom)

    return phi_polynoms


# CUSTOM PHI FUNCTIONS
def construct_custom_functions(phi_polynoms):
    """Creates a list of functions, that act like polynoms given by coefficients."""
    phi_functions = []

    for polynom in phi_polynoms:
        phi_functions.append(custom_phi(polynom))

    return phi_functions


def custom_phi(poly_coeffs):
    """Creates a phi function (polynom) with given coefficients."""
    def phi_function(point):
        return np.polyval(poly_coeffs, point)
    
    return phi_function


# PLOTTING
def polynomial_plot(points, values, approx_poly):
    x = np.linspace(min(points) - 0.5, max(points) + 0.5, 100)
    y = 0
    max_power = len(approx_poly)

    for power in range(max_power):
        y = y + approx_poly[power] * (x ** (max_power - power - 1))
    
    # size setting
    plt.figure(figsize=(10,6))

    # points
    plt.scatter(points, values, color="red", marker="x", s=200, linewidth=2, label="points")

    # curve
    plt.plot(x, y, color="blue", linewidth=2, label='approximation polynom')
    plt.legend()
    plt.show()