import unittest
import numpy as np
from main import generate_paillier_keys, paillier_encrypt, paillier_decrypt, matrix_factorization

class TestMainFunctions(unittest.TestCase):

    def test_generate_paillier_keys(self):
        private_key, public_key = generate_paillier_keys()
        self.assertIsNotNone(private_key)
        self.assertIsNotNone(public_key)

    def test_paillier_encrypt_decrypt(self):
        private_key, public_key = generate_paillier_keys()
        original_data = np.array([1, 2, 3], dtype=np.int64)
        encrypted_data = paillier_encrypt(original_data, public_key)
        decrypted_data = paillier_decrypt(encrypted_data, private_key)
        np.testing.assert_array_equal(original_data, decrypted_data)

    def test_matrix_factorization(self):
        R = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]], dtype=np.float64)
        N = len(R)
        M = len(R[0])
        K = 2
        P = np.random.rand(N, K)
        Q = np.random.rand(M, K)
        private_key, public_key = generate_paillier_keys()
        P, Q = matrix_factorization(R, P, Q, K, public_key, private_key)
        self.assertIsNotNone(P)
        self.assertIsNotNone(Q)


if __name__ == '__main__':
    unittest.main()
