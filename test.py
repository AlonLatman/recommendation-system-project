import logging
import unittest
import string
import random
from main import encrypt_data, decrypt_data

class TestEncryptionFunctions(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)

    def tearDown(self):
        self.logger.handlers.clear()

    def test_encrypt_decrypt(self):
        # Test encryption and decryption with random data
        original_data = b'This is a test data'
        participant_id = 1

        encrypted_data = encrypt_data(original_data, participant_id)
        decrypted_data = decrypt_data(encrypted_data, participant_id)

        self.assertEqual(original_data, decrypted_data)

    def test_iv(self):
        # Test if the IV remains the same after encryption and decryption
        original_data = b'This is a test data'
        participant_id = 1

        encrypted_data = encrypt_data(original_data, participant_id)

        # Extract the IV from the encrypted data
        iv = encrypted_data[:16]

        decrypted_data = decrypt_data(encrypted_data, participant_id)

        self.assertEqual(iv, encrypted_data[:16])

    def test_salt(self):
        # Test if the salt remains the same after encryption and decryption
        original_data = b'This is a test data'
        participant_id = 1

        encrypted_data = encrypt_data(original_data, participant_id)

        # Extract the salt from the encrypted data
        salt = encrypted_data[16:48]

        decrypted_data = decrypt_data(encrypted_data, participant_id)

        self.assertEqual(salt, encrypted_data[16:48])

    def test_empty_input(self):
        original_data = b''
        participant_id = 1

        encrypted_data = encrypt_data(original_data, participant_id)
        decrypted_data = decrypt_data(encrypted_data, participant_id)

        self.assertEqual(original_data, decrypted_data)

    def test_non_string_input(self):
        original_data = 123  # integer input
        participant_id = 1

        with self.assertRaises(TypeError):
            encrypted_data = encrypt_data(original_data, participant_id)

        original_data = None  # None input
        with self.assertRaises(TypeError):
            encrypted_data = encrypt_data(original_data, participant_id)

    def test_large_data(self):
        original_data = ''.join(random.choices(string.ascii_uppercase + string.digits, k=1000000)).encode()
        participant_id = 1

        encrypted_data = encrypt_data(original_data, participant_id)
        decrypted_data = decrypt_data(encrypted_data, participant_id)

        self.assertEqual(original_data, decrypted_data)

    def test_diff_participant_id(self):
        original_data = b'This is a test data'
        participant_id = 2

        encrypted_data = encrypt_data(original_data, participant_id)
        decrypted_data = decrypt_data(encrypted_data, participant_id)

        self.assertEqual(original_data, decrypted_data)


if __name__ == '__main__':
    unittest.main()
