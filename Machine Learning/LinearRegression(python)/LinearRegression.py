from numpy import *

def compute_error_for_line_given_points(b, m, points):
    # initialize error at zero
    totalError = 0
    # for every point
    for i in range(0, len(points)):
        # get x_i value
        x = points[i ,0]
        # get y_i value
        y = points[i, 1]
        # get the difference, square it and add it to the total
        totalError += (y - (m*x + b))**2

    # get the average
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # Starting b and m
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iterations):
        # update b and m with the new more accurate b and m by performing this gradient step
        b, m =step_gradient(b, m, array(points), learning_rate)
    
    return [b, m]

def step_gradient(b_current, m_current, points, learningRate):

    # Starting points for the gradient
    b_gradient = 0
    m_gradient = 0

    N = len(points)

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Direction with respect to b and m. We compute the partial 
        # We can calculate it analitically from error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x + b_current)))

    # Update b and m values with partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]


def run():

    # Step 1 - collect data (numpy)
    points = genfromtxt('data.csv', delimiter=',')

    # X from points contains the amount of hours studied and Y is the grade

    # Step 2 - define hyperparameters

    # How fast should our model converge? (shouldn't be too large because then it could not converge)
    learning_rate = 0.0001
    # y = m*x + b
    initial_b = 0
    initial_m = 0
    # How much do we wanna train this model
    num_iterations = 1000

    # Step 3 - train our model
    print('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print('Ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


if __name__ == '__main__':
    run()