from least_squares_method import least_squares, polynomial_plot

# 1)
points = [-2, -1, 0, 1, 2, 3]
values = [-15, -8, -7, -6, 1, 20]
weights = [1, 1, 1, 1, 1, 1]

# a)
approx_poly = least_squares(points, values, weights, poly_degree=1)
# approx_poly = least_squares(points, values, weights, poly_degree=4)

# b)
# weights = [1/8, 1/2, 1, 1, 1/2, 1/8]
# approx_poly = least_squares(points, values, weights, poly_degree=1)

# c)
# approx_poly = least_squares(points, values, weights, poly_degree=2)

# 2)

# points = [-3, -2, -1, 0, 1, 2, 3]
# values = [4, 2, 3, 0, -1, -2, -5]
# weights = [1, 1, 1, 1, 1, 1, 1]

# approx_poly = least_squares(points, values, weights, poly_degree=2)

# 3)

# points = [-1, -1/2, 0, 1/2, 1]
# values = [3, 1, 2, 7, 8]
# weights = [1, 1, 1, 1, 1]

# approx_poly = least_squares(points, values, weights, poly_degree=2, phi_polynoms=[[1], [1, 0], [2, 0, -1]])

# approx_poly = least_squares(points, values, weights, poly_degree=2)

print(approx_poly)
polynomial_plot(points, values, approx_poly)