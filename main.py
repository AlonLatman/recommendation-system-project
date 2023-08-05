import random
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tenseal as ts


context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40


def create_user_vectors(data):
    """Create a vector for each user representing their ratings."""
    # Pivot the data to create a user-item matrix
    user_item_matrix = data.pivot(index='User_ID', columns='Item_ID', values='Rating')

    # Replace missing values with 0
    user_item_matrix.fillna(0, inplace=True)

    # Convert the user-item matrix into a list of user vectors
    user_vectors = [list(row) for row in user_item_matrix.values]

    return user_vectors


def cosine_similarity(encrypted_vector1, encrypted_vector2, norm1, norm2):
    """Calculate the cosine similarity between two encrypted vectors."""
    # Calculate the dot product between the vectors
    dot_product = encrypted_vector1.dot(encrypted_vector2)

    # Decrypt the dot product
    decrypted_dot_product = decrypt_vector(dot_product)

    # Calculate the cosine similarity
    similarity = decrypted_dot_product / (norm1 * norm2)

    return similarity


def calculate_similarities(user_id, encrypted_vectors, norms):
    """Calculate the similarities between a given user's vector and all user vectors."""
    # Get the encrypted vector and norm for the given user
    encrypted_vector1 = encrypted_vectors[user_id]
    norm1 = norms[user_id]

    # Calculate the similarity between the given user's vector and each user vector
    similarities = []
    for i, encrypted_vector2 in enumerate(encrypted_vectors):
        norm2 = norms[i]
        similarity = cosine_similarity(encrypted_vector1, encrypted_vector2, norm1, norm2)
        similarities.append(similarity)

    return similarities


def find_similar_users(user_id, similarities, n=10):
    """Find the users who are most similar to a given user."""
    # Create a list of user IDs and their corresponding similarities
    user_similarities = list(enumerate(similarities))

    # Sort the list by similarity in descending order
    user_similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the user IDs of the most similar users
    similar_users = [user_id for user_id, similarity in user_similarities[:n]]

    return similar_users


def load_data_from_excel(file_path):
    # Load the data from the Excel file
    df = pd.read_excel(file_path)

    # Process the data and create a user-item matrix
    user_item_matrix = df.pivot(index='User_ID', columns='Item_ID', values='Rating')
    user_item_matrix.fillna(0, inplace=True)
    data = user_item_matrix.to_numpy(dtype=np.float32)

    return data


def encrypt_vector(vector, context):
    """Encrypt a vector using the CKKS scheme."""
    encrypted_vector = ts.ckks_vector(context, vector)
    return encrypted_vector


def decrypt_vector(encrypted_vector):
    """Decrypt a vector encrypted with the CKKS scheme."""
    decrypted_vector = encrypted_vector.decrypt()
    return decrypted_vector


def recommend_items(user_index, similar_users_indices, data, n=10):
    """Recommend items to a user based on the ratings of similar users."""
    # Get the User_ID of the given user
    user_id = data['User_ID'].unique()[user_index]

    # Get the items that the given user has already rated
    user_items = set(data[data['User_ID'] == user_id]['Item_ID'])

    # Create a dictionary to count the recommendations for each item
    recommendation_counts = {}

    # For each similar user, find the items they have rated highly
    for similar_user_index in similar_users_indices:
        # Get the User_ID of the similar user
        similar_user_id = data['User_ID'].unique()[similar_user_index]

        similar_user_highly_rated_items = set(
            data[(data['User_ID'] == similar_user_id) & (data['Rating'] >= 4)]['Item_ID']
        )

        # Subtract the items that the given user has already rated
        recommended_items = similar_user_highly_rated_items - user_items

        # Add the recommended items to the recommendation counts
        for item in recommended_items:
            if item in recommendation_counts:
                recommendation_counts[item] += 1
            else:
                recommendation_counts[item] = 1

    # Sort the items by recommendation count in descending order
    recommended_items_sorted = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)

    # Get the items with the highest recommendation counts
    top_recommended_items = [item for item, count in recommended_items_sorted[:n]]

    return top_recommended_items


def sum_ratings(encrypted_data, private_key):
    # Calculate the sum of the ratings for each item
    sums = np.empty(encrypted_data.shape[1], dtype=object)
    for j in range(encrypted_data.shape[1]):
        # Initialize the sum to an encrypted 0
        sums[j] = encrypted_data[0, j] - encrypted_data[0, j]
        for i in range(encrypted_data.shape[0]):
            # Add the encrypted rating to the sum
            sums[j] += encrypted_data[i, j]

    # Decrypt the sums
    decrypted_sums = [private_key.decrypt(sum) for sum in sums]

    return decrypted_sums


def generate_synthetic_data(participant_count: object, items_per_participant: object) -> object:
  data = []

  for participant_id in range(1, participant_count + 1):
    participant_data = []
    for item_id in range(1, items_per_participant + 1):
      user_id = random.randint(1000, 9999)  # Generate random user ID
      rating = random.randint(1, 5)  # Generate random rating

      participant_data.append({'User_ID': user_id, 'Item_ID': item_id, 'Rating': rating})

    data.extend(participant_data)

    # Create a separate Excel file for each participant
    participant_df = pd.DataFrame(participant_data)
    excel_filename = f'participant_{participant_id}_data.xlsx'
    participant_df.to_excel(excel_filename, index=False)
    print(f'Synthetic data for Participant {participant_id} saved to {excel_filename}')

  # Create a combined Excel file for all participants
  combined_df = pd.DataFrame(data)
  combined_excel_filename = 'combined_participant_data.xlsx'
  combined_df.to_excel(combined_excel_filename, index=False)
  print(f'Combined synthetic data for all participants saved to {combined_excel_filename}')


# def main():
#     # Open a file dialog for the user to select the Excel file
#     root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     generate_synthetic_data(participant_count=3, items_per_participant=5)
#     file_path = filedialog.askopenfilename()
#
#     if file_path:  # Check that a file was selected
#         # Load the data
#         data = pd.read_excel(file_path)
#
#         # Create user vectors
#         user_vectors = create_user_vectors(data)
#
#         # Create a TenSEAL context for the encryption
#         context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
#         context.global_scale = 2 ** 40
#         context.generate_galois_keys()
#
#         # Encrypt the user vectors and calculate their norms
#         encrypted_vectors = []
#         norms = []
#         for vector in user_vectors:
#             encrypted_vectors.append(encrypt_vector(vector, context))
#             norms.append(np.linalg.norm(vector))
#
#         # Choose a user to make recommendations for
#         user_id = random.randint(0, 2)
#
#         # Calculate the similarities
#         similarities = calculate_similarities(user_id, encrypted_vectors, norms)
#
#         # Find the most similar users
#         similar_users = find_similar_users(user_id, similarities)
#
#         # Recommend items
#         recommendations = recommend_items(user_id, similar_users, data)
#
#         if len(recommendations) > 0:
#             print("Recommended items:", recommendations)
#         else:
#             print("Recommended items returns empty there are no Recommended items with high rating for this user")
#
#
# main()