"""
MT-19937
Mersenne Twister PRNG
reference: https://en.wikipedia.org/wiki/Mersenne_Twister
"""


import numpy as np


class Mersenne:
    def __init__(self):
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253

        self.lower_mask = 0x7FFFFFFF
        self.upper_mask = 0x80000000

        self.index = self.n + 1

        self.x = np.zeros((self.n,), dtype=np.uint32)

    def seed_mt(self, seed):
        self.x[0] = seed
        self.index = self.n
        for i in range(1, self.n):
            self.x[i] = self.f * (self.x[i - 1] ^ (self.x[i - 1] >> (self.w - 2))) + i

    def twist(self):
        for i in range(self.n):
            _x = (self.x[i] & self.upper_mask) + (
                self.x[(i + 1) % self.n] & self.lower_mask
            )
            xA = _x >> 1
            if (_x % 2) != 0:
                xA = xA ^ self.a
            self.x[i] = self.x[(i + self.m) % self.n] ^ xA

        self.index = 0

    def extract_number(self):
        if self.index == self.n:
            self.twist()

        y = self.x[self.index]
        y ^= (y >> self.u) & self.d
        y ^= (y >> self.s) & self.b
        y ^= (y >> self.t) & self.c
        y ^= y >> 1
        self.index += 1
        return y & ((1 << self.w) - 1)


# ================ TESTS ===========================
"""
Frequency test as specified in NIST SP 800-22
count the no of zeros and ones in bit sequences
they should be very close
"""


def freq_bit_test():
    r = Mersenne()
    r.seed_mt(100)

    # test on 10000 values
    values = np.array(
        [r.extract_number() for _ in range(10000)],
        dtype=np.uint32,
    )

    cnt = 0
    for num in values:
        cnt += 2 * num.bit_count() - 32

    test_statistic = abs(cnt) / np.sqrt(10000 * 32)
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

    r = Mersenne()
    r.seed_mt(100)

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
