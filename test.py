import logging
import unittest
import random
from io import StringIO
import numpy as np
from main import (
    generate_encryption_key,
    apply_differential_privacy,
    generate_shares,
    train_model,
    recommend_items
)


class TestEncryptionFunctions(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        self.logger.addHandler(handler)

    def tearDown(self):
        self.logger.handlers.clear()

    def test_generate_encryption_key(self):
        print("Running test_generate_encryption_key...")
        for _ in range(5):
            key1 = generate_encryption_key()
            key2 = generate_encryption_key()
            self.assertNotEqual(key1, key2)  # Ensure keys are different each time

        # Negative test: Check for invalid key_size (key_size should be a positive integer)
        print("Running negative test_generate_encryption_key...")
        with self.assertRaises(TypeError):
            invalid_key_size = generate_encryption_key(key_size=7)

        print("Test Passed!")

    def test_apply_differential_privacy(self):
        print("Running test_apply_differential_privacy...")

        # Positive test: Check if the shape of private data is the same as the original data
        for _ in range(5):
            data = np.array([1, 2, 3, 4])
            epsilon = 0.1
            private_data = apply_differential_privacy(data, epsilon)
            self.assertEqual(data.shape, private_data.shape)

        # Negative test: Check for invalid data type (data should be a numpy array)
        print("Running negative test_apply_differential_privacy...")
        invalid_data = [1, 2, 3, 4]  # Invalid data type (list instead of numpy array)
        epsilon = 0.1
        with self.assertRaises(TypeError):
            apply_differential_privacy(invalid_data, epsilon)

        print("Test Passed!")

    def test_generate_shares(self):
        print("Running test_generate_shares...")

        # Positive test: Check if the number of shares is greater than or equal to the threshold
        for _ in range(5):
            data = np.array([1, 2, 3, 4])
            threshold = 2
            shares = generate_shares(data, threshold)
            self.assertTrue(threshold <= shares.shape[0])

        # Negative test: Check for invalid threshold (threshold should be greater than 0)
        print("Running negative test_generate_shares...")
        with self.assertRaises(ValueError):
            data = np.array([1, 2, 3, 4])
            invalid_threshold = -1
            generate_shares(data, invalid_threshold)

        print("Test Passed!")

    def test_train_model(self):
        print("Running test_train_model...")
        for _ in range(2):
            shares = np.random.random((10, 10))
            model = train_model(shares)
            self.assertTrue(np.array_equal(shares.shape, model.shape))
        print("Test Passed!")

    def test_recommend_items(self):
        print("Running test_recommend_items...")

        # Positive test: Check for recommended items with known user_id
        model = np.random.random((10, 10))
        user_id = 2
        recommended_items1 = recommend_items(model, user_id)

        # Re-run with the same user_id to check consistency
        user_id = 2
        recommended_items2 = recommend_items(model, user_id)
        self.assertTrue(np.array_equal(recommended_items1, recommended_items2))

        # Negative test: Check for invalid user_id (user_id should be within the model's shape)
        invalid_user_id = model.shape[0] + 1  # Invalid user_id (outside the model's shape)
        with self.assertRaises(ValueError):
            recommend_items(model, invalid_user_id)

        # Negative test: Check for invalid model shape (model should have at least two dimensions)
        invalid_model = np.random.random(5)  # Invalid model shape (1D array)
        with self.assertRaises(ValueError):
            recommend_items(invalid_model, user_id)

        self.log_capture.seek(0)
        logs = self.log_capture.read()
        self.assertEqual(logs, '', msg="Log messages with level WARNING or higher were captured.")
        print("Test Passed!")


if __name__ == '__main__':
    unittest.main()