import math
import numpy
import matplotlib.pyplot as plt
import itertools
import random

dataset = [(19, 85, 'red'), (18, 79, 'blue'), (85, 89, 'blue'), (44, 46, 'red'), (42, 4, 'blue'), (49, 27, 'red'), (22, 44, 'red'), (83, 99, 'red'), (59, 9, 'blue'), (55, 49, 'blue'), (23, 85, 'blue'), (63, 84, 'red'), (55, 83, 'red'), (11, 11, 'red'), (69, 9, 'blue'), (30, 74, 'red'), (26, 87, 'blue'), (79, 11, 'red'), (55, 46, 'blue'), (6, 17, 'red'), (18, 7, 'blue'), (80, 72, 'blue'), (79, 99, 'red'), (29, 65, 'red'), (90, 65, 'blue'), (97, 55, 'red'), (63, 86, 'red'), (70, 31, 'red'), (13, 9, 'red'), (47, 17, 'blue'), (74, 55, 'blue'), (73, 26, 'blue'), (66, 11, 'blue'), (21, 23, 'blue'), (30, 69, 'red'), (72, 60, 'blue'), (26, 25, 'red'), (82, 46, 'blue'), (45, 18, 'blue'), (57, 10, 'red'), (17, 88, 'blue'), (42, 1, 'blue'), (63, 97, 'blue'), (98, 48, 'blue'), (75, 32, 'red'), (25, 80, 'red'), (52, 34, 'blue'), (100, 22, 'blue'), (24, 24, 'blue'), (87, 90, 'red'), (7, 58, 'red'), (47, 70, 'red'), (95, 37, 'blue'), (72, 52, 'red'), (8, 23, 'blue'), (42, 70, 'red'), (20, 3, 'blue'), (49, 99, 'blue'), (52, 40, 'blue'), (15, 67, 'blue'), (56, 36, 'red'), (89, 28, 'red'), (73, 69, 'red'), (68, 49, 'blue'), (27, 8, 'blue'), (28, 44, 'blue'), (100, 86, 'red'), (22, 77, 'blue'), (73, 100, 'red'), (4, 46, 'red')]
validation_set = [(71, 71, 'blue'), (78, 65, 'blue'), (21, 64, 'blue'), (82, 53, 'blue'), (19, 46, 'blue'), (9, 35, 'blue'), (85, 73, 'blue'), (20, 24, 'blue'), (87, 19, 'blue'), (90, 16, 'blue'), (88, 10, 'blue'), (97, 85, 'red'), (85, 77, 'red'), (42, 56, 'red'), (67, 77, 'blue'), (11, 97, 'red'), (69, 35, 'blue'), (94, 62, 'blue'), (19, 61, 'blue'), (44, 76, 'red'), (98, 84, 'blue'), (97, 22, 'blue'), (32, 56, 'blue'), (60, 20, 'blue'), (92, 70, 'red'), (92, 94, 'blue'), (87, 21, 'red'), (20, 29, 'red'), (19, 97, 'blue'), (25, 4, 'blue')]
test_set = [(30, 55, 'blue'), (63, 60, 'red'), (100, 19, 'blue'), (32, 83, 'red'), (65, 68, 'blue'), (83, 19, 'blue'), (1, 54, 'blue'), (61, 80, 'blue'), (99, 75, 'red'), (81, 7, 'blue'), (99, 81, 'red'), (6, 16, 'blue'), (47, 1, 'red'), (14, 61, 'blue'), (65, 29, 'blue'), (45, 90, 'blue'), (89, 59, 'red'), (2, 75, 'blue'), (59, 53, 'red'), (59, 80, 'red'), (16, 2, 'red'), (6, 2, 'red'), (19, 48, 'blue'), (15, 88, 'red'), (83, 30, 'red'), (82, 48, 'blue'), (89, 54, 'blue'), (6, 62, 'red'), (73, 93, 'red'), (61, 74, 'red')]


def distt(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    return the distance between 2 points
    :param x1, y1: x and y coordinates of first point
    :param x2, y2: x and y coordinates of second point
    :return: float distance
    """
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def check_prediction_for_point(K: int, x_coordinate: int, y_coordinate: int, expected_color: str) -> bool:
    """
    find the K nearest neighbors for a given x, y and return True if the
    decided color equals to the expected_color
    :param K: the number of neighbors to check
    :param x_coordinate: x coordinate of the point to check
    :param y_coordinate: y coordinate of the point to check
    :param expected_color: expected color of the point
    :return: True if expected_color == prediction and False otherwise
    """
    predicted_color = None
    dict_of_dst = {}

    for x, y, color in dataset:
        dict_of_dst[(x, y, color)] = distt(x, y, x_coordinate, y_coordinate) # making a dict (key: point, value: dist from the param (x,y))

    dict_of_dst = dict(sorted(dict_of_dst.items(), key=lambda item: item[1])) # sorting
    first_K = list(dict_of_dst.keys())[:K] # taking the first K data points from the sorted dict

    colors_count = {}

    # counting how many points there are in every color
    for _, _, color in first_K:
        if color not in colors_count.keys():
            colors_count[color] = 0
        
        colors_count[color] += 1
        
    groups = itertools.groupby(colors_count.values())
    next(groups, None)
    
    if next(groups, None) is None:
        # All values are equal.
        colors_avg = {} # the K first colors and the avg - {'color': [sum, number of dists]}
        count = 1

        try:
            # making avg of each color from the first K points
            for (_, _, color), dist in dict_of_dst.items():
                if color not in colors_avg.keys():
                    colors_avg[color] = [0, 0]
                
                colors_avg[color][0] += dist
                colors_avg[color][1] += 1

                if count == K:
                    break
                count += 1
                
            for color, avg_params in colors_avg.items():
                colors_avg[color] = avg_params[0] / avg_params[1]

            equal = True
            perv_avg = list(dict_of_dst.values())[0]
            
            # checking if all the avgs are equal
            for _, avg in colors_avg.items():
                if avg != perv_avg:
                    equal = False
                    break
                
            # if they are all equal - picking random one
            if equal:
                predicted_color = random.choice(list(colors_avg.keys()))
            
            # else - taking the color with the least avg dist
            else:
                colors_avg = dict(sorted(colors_avg.items(), key=lambda item: item[1]))
                predicted_color = list(colors_avg)[0] # taking the first key from sorted dict

        except IndexError:
            print("Oops. K is bigger than the amount of points in the dataset\n")
    
    else:
        # Unequal values detected.
        colors_count = dict(sorted(colors_count.items(), key=lambda item: item[1])[::-1]) # sorting from greater to smaller
        predicted_color = list(colors_count.keys())[0]

    return predicted_color == expected_color


def cnt_error_validation_set(K: int) -> int:
    """
    Go over the validation set and count for errors in evaluation
    :param K: K to check
    :return: number of errors in evaluation of validation set
    """
    cnt_error = 0
    for x, y, expected_color in validation_set:
        if not check_prediction_for_point(K, x, y, expected_color):
            cnt_error += 1

    return cnt_error


def run_KNN() -> int:
    """
    Find the best K for the K-NN algorithm
    :return: the best K (integer between 1 and 25)
    """
    best_k = 1
    errors = cnt_error_validation_set(best_k)
    
    for k in range(2,25):
        curr_errors = cnt_error_validation_set(k)

        if curr_errors < errors:
            errors = curr_errors
            best_k = k


    print("Best K is: {}\n".format(best_k))
    return best_k


def plot_dataset():
     # Nothing to change in this function! (Plot the point)
    for (x, y, color) in dataset:
        plt.scatter(x, y, color=color)
    plt.xlabel("Wealth")
    plt.ylabel("Religiousness")
    plt.show()


def run_test_set(K):
    """
    Go over the test set and count for errors in evaluation
    :param K: K to check
    :return: number of errors in evaluation of validation set
    """
    cnt_error = 0
    for x, y, expected_color in test_set:
        if not check_prediction_for_point(K, x, y, expected_color):
            cnt_error += 1
    print(f"Number of errors for K={K} on test set: {cnt_error}")
    print(f"Success rate: {int((len(test_set)-cnt_error)/ len(test_set) * 100)}%")


if __name__ == "__main__":
    plot_dataset()  # Show dataset (only for your convenience)
    best_K = run_KNN()  # find the best K by running over the validation set
    run_test_set(best_K)  # calc the results for the test set

