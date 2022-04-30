"""
Linear congruential generator
reference: https://en.wikipedia.org/wiki/Linear_congruential_generator
"""
import numpy as np


class Lcg:
    def __init__(self):
        self.a = 22695477
        self.c = 12345
        self.m = 1 << 32

    def seed_lcg(self, seed):
        self.rand = seed

    def extract_number(self):
        self.rand = (self.a * self.rand + self.c) % self.m
        return self.rand


# ================ TESTS ===========================
"""
Frequency test as specified in NIST SP 800-22
count the no of zeros and ones in bit sequences
they should be very close
"""


def freq_bit_test():
    r = Lcg()
    r.seed_lcg(214843063)

    # test on 10000 values
    values = np.array(
        [r.extract_number() for _ in range(10000)],
        dtype=np.uint32,
    )

    cnt = 0
    for num in values:
        cnt += 2 * num.bit_count() - 32

    test_statistic = abs(cnt) / np.sqrt(1000 * 32)
    p_value = np.math.erfc(test_statistic / np.sqrt(2))

    print("bit frequency test:", end=" ")
    if p_value < 0.01:
        print("sequence is not random")
    else:
        print("sequence is random")


"""
Cumulative sum test as specified in NIST SP 800-22
"""


def cumulative_sum_test():
    from scipy.stats import norm

    r = Lcg()
    r.seed_lcg(214843063)

    values = np.array(
        [r.extract_number() for _ in range(10000)],
        dtype=np.uint32,
    )

    binary_data = ""
    for num in values:
        binary_data += bin(num)[2:].zfill(32)

    length_bits = len(binary_data)
    counts = np.zeros((length_bits,))

    for counter, char in enumerate(binary_data):
        sub = 1
        if char == "0":
            sub = -1
        if counter > 0:
            counts[counter] = counts[counter - 1] + sub
        else:
            counts[counter] = sub

    z = max(abs(counts))

    lower_bound = int(0.25 * np.floor(-length_bits / z) + 1)
    upper_bound = int(0.25 * np.floor(length_bits / z) - 1)

    terms_one = []
    for k in range(lower_bound, upper_bound + 1):
        sub = norm.cdf((4 * k - 1) * z / np.sqrt(length_bits))
        terms_one.append(norm.cdf((4 * k + 1) * z / np.sqrt(length_bits)) - sub)

    lower_bound = int(np.floor(0.25 * np.floor(-length_bits / z - 3)))
    upper_bound = int(np.floor(0.25 * np.floor(length_bits / z) - 1))

    terms_two = []
    for k in range(lower_bound, upper_bound + 1):
        sub = norm.cdf((4 * k + 1) * z / np.sqrt(length_bits))
        terms_two.append(norm.cdf((4 * k + 3) * z / np.sqrt(length_bits)) - sub)

    p_value = 1.0 - sum(np.array(terms_one)) + sum(np.array(terms_two))

    print("cumulative sum test:", end=" ")
    if p_value < 0.01:
        print("sequence is not random")
    else:
        print("sequence is random")


freq_bit_test()
cumulative_sum_test()
