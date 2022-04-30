"""
EL Gamal simulation

consider a situation where Bob wishes to send a message to Alice

steps to send message using the Elgamal cryptosystem

1. Bob generates his private key and public key pair.
2. Bob communicates the public key to Alice.
3. Alice uses Bob's public to encrypt the message by choosing a random value called ephemeral key.
4. Alice sends a pair of messages to Bob.
5. Bob decrypts the message using his private key.

reference: https://en.wikipedia.org/wiki/ElGamal_encryption
"""

from gmpy2 import next_prime
import random


def get_private_key(n=1024):
    p = next_prime(random.getrandbits(n))
    g = random.randint(2, p - 1)
    x = random.randint(2, p - 1)
    h = pow(g, x, p)

    return (p, g, x, h)


def encrypt_message(p, g, h, msg):
    y = random.randint(2, p - 1)
    s = pow(h, y, p)
    c1 = pow(g, y, p)
    c2 = (msg * s) % p

    return (c1, c2)


def decrypt_message(p, x, c1, c2):
    s = pow(c1, x, p)
    msg = (c2 * pow(s, -1, p)) % p
    return msg


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, "big")


if __name__ == "__main__":
    # Bob side
    (p, g, x, h) = get_private_key()

    # Alice side
    # Alice recieves p, g and h
    data = b"cryptography is love"
    msg = int.from_bytes(data, "big")
    c1, c2 = encrypt_message(p, g, h, msg)

    # Bob side
    # Bob recieves c1, c2 from Alice
    recv_msg = decrypt_message(p, x, c1, c2)
    recv_data = int_to_bytes(int(recv_msg))
    assert recv_data == data
    print(recv_data)
