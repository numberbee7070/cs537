import numpy as np
from des import *
import random
from matplotlib import pyplot as plt


def hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> int:
    """
    returns hamming distance between two given arrays
    """
    return np.count_nonzero(arr1 != arr2)


def encrypt(key: list, data: list):
    """
    this function will take key and data as list of binary bits
    and return an array of 16 values of each round
    """

    _original_data = data  # to compute HD with original data
    round_val = []  # data to return

    # generate subkeys
    key16 = keyschedule(key)

    # initial permutation of data
    data = permutation(data, 0)

    # round function
    for i in range(16):
        data = round(data, key16[i])
        round_val.append(data)

    # making left side right and right side left
    data = np.roll(data, 32)

    # final encrypted output
    data = permutation(data, 1)

    return round_val


def get_bit_array(x: int):
    """
    returns a sequence of random bits of length x
    """
    return np.array(random.choices((0, 1), k=x))


def flip_n_bits(arr: np.ndarray, n: int):
    """
    flip n random bits in given bit string
    to create a hamming distance of n
    """

    for x in random.sample(range(len(arr)), n):
        arr[x] = 1 - arr[x]

    return arr


def experiment1():
    """
    choose a random plaintext and a key
    take 5 plaintexts with different hamming distances from original
    encrypt all with DES with same key
    plot the array of hamming distances at each round
    """

    key = get_bit_array(56)
    pt = get_bit_array(64)

    original_rounds = encrypt(key, pt)

    # iterate over 1 to 5 hamming distances and compare with original round values
    for ham in range(1, 6):
        # this is new plaintext with specified hamming distance from initial plaintext
        new_pt = flip_n_bits(pt, ham)

        # round values of new plaintext
        new_rounds = encrypt(key, new_pt)

        # array of hamming distances
        ham_dist_arr = []

        # compute hamming distance at each round value
        for a, b in zip(original_rounds, new_rounds):
            ham_dist_arr.append(hamming_distance(a, b))

        # plot hamming distances of all 5 plaint texts on same graph
        plt.plot(list(range(1, 17)), ham_dist_arr, label=f"{ham}")

    plt.legend(title="Hamming distance with original plain text")
    plt.show()


def experiment2():
    """
    choose a random plaintext and a key
    take 5 keys with different hamming distances from original
    encrypt same plaintext with DES with different keys
    plot the array of hamming distances at each round
    """

    key = get_bit_array(56)
    pt = get_bit_array(64)

    original_rounds = encrypt(key, pt)

    # iterate over 1 to 5 hamming distances and compare with original round values
    for ham in range(1, 6):
        # this is new key with specified hamming distance from initial key
        new_key = flip_n_bits(key, ham)

        # round values with new key
        new_rounds = encrypt(new_key, pt)

        # array of hamming distances
        ham_dist_arr = []

        # compute hamming distance at each round value
        for a, b in zip(original_rounds, new_rounds):
            ham_dist_arr.append(hamming_distance(a, b))

        # plot hamming distances of all 5 plaint texts on same graph
        plt.plot(list(range(1, 17)), ham_dist_arr, label=f"{ham}")

    plt.legend(title="Hamming distance with original key")
    plt.show()


experiment1()
experiment2()
