import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming your main.py has all the necessary functions
import main

class TestRecommendationSystem(unittest.TestCase):
    def setUp(self):
        # Load the data
        self.data = pd.read_excel('C:\\Users\\Latman\\PycharmProjects\\pythonProject11\\combined_participant_data.xlsx')

        # Split the data into training and test sets
        self.training_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)

    def test_recommendations(self):
        # Try to get a user from the test set that also exists in the training set
        test_user_id = None
        for uid in self.test_data['User_ID'].unique():
            if uid in self.training_data['User_ID'].unique():
                test_user_id = uid
                break

        # # If no such user is found after the loop, the test should fail
        # if test_user_id is None:
        #     self.fail("No test user found in the training data.")

        # Get the index of the test user in the training data
        user_indices = self.training_data['User_ID'].unique().tolist()
        test_user_index = user_indices.index(test_user_id) if test_user_id in user_indices else None

        # Get the actual items rated by the test user
        actual_items = set(self.test_data[self.test_data['User_ID'] == test_user_id]['Item_ID'])

        # We would also need to retrieve the encrypted vectors and norms for our users.
        user_vectors = main.create_user_vectors(self.training_data)
        encrypted_vectors = [main.encrypt_vector(vector, main.context) for vector in user_vectors]
        norms = [np.linalg.norm(vector) for vector in user_vectors]

        # Calculate similarities for the test user
        similarities = main.calculate_similarities(test_user_index, encrypted_vectors, norms)

        # Find the top similar users to our test user
        similar_users_indices = main.find_similar_users(test_user_index, similarities, n=10)

        # Make recommendations for the test user using the training data
        recommended_items = set(main.recommend_items(test_user_index, similar_users_indices, self.training_data, n=10))

        # Calculate precision: the proportion of recommended items that are relevant
        if recommended_items:
            precision = len(recommended_items & actual_items) / len(recommended_items)
        else:
            precision = 0

        # Check if the precision is above a certain threshold (e.g., 0.1)
        self.assertGreaterEqual(precision, 0.1, "The precision of the recommendations is below the threshold.")


if __name__ == '__main__':
    unittest.main()
