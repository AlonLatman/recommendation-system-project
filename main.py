import logging  # Importing the logging module for generating logging messages and debugging
import os
import random  # Importing the random module for generating random numbers and selections

import joblib
import pandas as pd  # Importing pandas, a powerful library for data manipulation and analysis
import numpy as np  # Importing numpy for numerical operations and working with arrays
import tenseal as ts  # Importing tenseal for homomorphic encryption operations
import tensorflow as tf
from keras import Model
from keras.src.layers import Dot, Flatten, Embedding
from pyexpat import model
from sklearn.ensemble import IsolationForest

# global context for encryption
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40


def create_user_vectors(data):
    """
    Create a vector for each user representing their ratings for different items. The function creates a user-item matrix
    from the data, replaces missing values with 0 (assuming the user has not rated that item), and then converts the
    matrix into a list of user vectors.

    Parameters:
        data (pd.DataFrame): The data frame that contains the ratings of the users. It should have columns named 'User_ID',
                         'Item_ID', and 'Rating'.

    Returns:
        list: A list of user vectors. Each vector is a list of ratings corresponding to the items. The index in the outer
          list corresponds to the user's ID.

    Example:
        >>>create_user_vectors(data)
        will return a list of user vectors based on the ratings in the data frame.
    """
    # Pivot the data to create a user-item matrix
    user_item_matrix = data.pivot(index='User_ID', columns='Item_ID', values='Rating')

    # Replace missing values with 0
    user_item_matrix.fillna(0, inplace=True)

    # Convert the user-item matrix into a list of user vectors
    user_vectors = [list(row) for row in user_item_matrix.values]

    return user_vectors


def cosine_similarity(encrypted_vector1, encrypted_vector2, norm1, norm2):
    """
    Calculate the cosine similarity between two encrypted vectors. Cosine similarity is a measure of similarity between
    two non-zero vectors, defined to equal the cosine of the angle between them. The function requires the norms of the
    vectors, as well as a decryption function for the dot product of the encrypted vectors.

    Parameters:
        encrypted_vector1 (list): The first encrypted vector.
        encrypted_vector2 (list): The second encrypted vector.
        norm1 (float): The norm (magnitude) of the first encrypted vector.
        norm2 (float): The norm (magnitude) of the second encrypted vector.

    Returns:
        float: The cosine similarity between the two encrypted vectors.

    Example:
        >>>cosine_similarity([1, 0], [0, 1], 1.0, 1.0)
        will return 0.0 because the vectors are orthogonal.
    """
    # Calculate the dot product between the vectors
    dot_product = encrypted_vector1.dot(encrypted_vector2)

    # Decrypt the dot product
    decrypted_dot_product = decrypt_vector(dot_product)

    # Calculate the cosine similarity
    similarity = decrypted_dot_product / (norm1 * norm2)

    return similarity


def calculate_similarities(user_id, encrypted_vectors, norms):
    """
    Calculate the similarities between a given user's vector and all other user vectors. The similarity is calculated
    using the cosine similarity, which measures the cosine of the angle between two vectors.

    Parameters:
    user_id (int): The ID of the user for whom we want to calculate similarities.
    encrypted_vectors (list): A list of encrypted vectors corresponding to each user. The index in the list corresponds
                              to the user's ID.
    norms (list): A list of norms corresponding to each user's encrypted vector. The index in the list corresponds to
                  the user's ID.

    Returns:
    list: A list of similarity scores corresponding to each user. The index in the list corresponds to the user's ID.

    Example:
        >>>calculate_similarities(0, encrypted_vectors, norms)
        will return a list of similarity scores between the user at index 0 and all other users,
        based on their encrypted vectors and norms.
    """
    if not isinstance(user_id, int) or user_id < 0 or user_id >= len(encrypted_vectors) or user_id >= len(norms):
        raise ValueError(
            f"user_id should be a non-negative integer less than the length of encrypted_vectors and norms. Got: {user_id}")

    if not isinstance(encrypted_vectors, list) or len(encrypted_vectors) == 0:
        raise ValueError("encrypted_vectors should be a non-empty list.")

    if not isinstance(norms, list) or len(norms) == 0 or len(norms) != len(encrypted_vectors):
        raise ValueError("norms should be a non-empty list with the same length as encrypted_vectors.")

    # Continue with the original function logic
    encrypted_vector1 = encrypted_vectors[user_id]
    norm1 = norms[user_id]

    similarities = []
    for i, encrypted_vector2 in enumerate(encrypted_vectors):
        norm2 = norms[i]
        try:
            similarity = cosine_similarity(encrypted_vector1, encrypted_vector2, norm1, norm2)
            similarities.append(similarity)
        except Exception as e:
            # Handle any potential errors arising from the cosine_similarity function
            print(f"Error calculating similarity for user ID {i}. Error: {e}")
            similarities.append(0)  # Default to 0 similarity for error cases

    return similarities


def find_similar_users(user_id, similarities, n=10):
    """
    Identifies the indices of the top n users most similar to the given user based on similarity scores.

    Parameters:
        user_id (int): ID of the target user.
        similarities (list[float]): List of similarity scores, where the index corresponds to the user's index.
        n (int, optional): Number of similar users to retrieve. Defaults to 10.

    Returns:
        list[int]: Indices of the top n most similar users.

    Raises:
        ValueError: If user_id is out of bounds, similarities is not a valid list, or n is not a valid count.

    Example:
        >>> find_similar_users(0, [0.1, 0.3, 0.2])
        This will output [1, 2], indicating the indices of users most similar to the user at index 0.

    Note:
        Similarity scores should be in the range [0, 1] where a higher value indicates greater similarity.
    """

    if not isinstance(user_id, int) or user_id < 0 or user_id >= len(similarities):
        raise ValueError(
            f"user_id should be a non-negative integer less than the length of similarities. Got: {user_id}")

    if not isinstance(similarities, list) or len(similarities) == 0:
        raise ValueError("similarities should be a non-empty list.")

    if not (isinstance(n, int) and 0 < n <= len(similarities)):
        raise ValueError(f"n should be a positive integer not exceeding the length of similarities. Got: {n}")

    # Continue with the original function logic
    user_similarities = list(enumerate(similarities))
    user_similarities.sort(key=lambda x: x[1], reverse=True)
    similar_users = [user_idx for user_idx, similarity in user_similarities[:n]]

    return similar_users


def load_data_from_excel(file_path):
    """
    Loads data from an Excel file and processes it to create a user-item matrix.

    Parameters:
        file_path (str): The path to the Excel file that contains the data.

    Returns:
        numpy.ndarray: A 2D numpy array (user-item matrix) where rows represent users, columns represent items,
        and the values represent ratings.
        Unrated items are represented as 0.

    Description:
        This function loads data from a given Excel file and processes it to create a user-item matrix.
        The input Excel file is expected to have at least the following columns: User_ID, Item_ID, and Rating.

    The function reads the Excel file into a DataFrame and then pivots the DataFrame to create a user-item matrix.
    Each row of the matrix corresponds to a user, each column corresponds to an item, and the matrix values represent
    the ratings given by users to items.
    If a user hasn't rated an item, the corresponding matrix value is filled with 0.

    Assumptions:
        The Excel file contains columns named User_ID, Item_ID, and Rating.
        The Rating column contains numeric values.
        The pandas and numpy libraries have been imported as pd and np respectively.
    """
    # Load the data from the Excel file
    df = pd.read_excel(file_path)

    # Process the data and create a user-item matrix
    user_item_matrix = df.pivot(index='User_ID', columns='Item_ID', values='Rating')
    user_item_matrix.fillna(0, inplace=True)
    data = user_item_matrix.to_numpy(dtype=np.float32)

    return data


def encrypt_vector(vector, context):
    """
    Encrypts a vector using the CKKS encryption scheme.

    Parameters:
        vector (list): List of values to be encrypted. Must be non-empty.
        context: Context required by the CKKS encryption scheme.

    Returns:
        Encrypted version of the input vector.

    Raises:
        ValueError: If vector is not a list or is empty.
        Exception: If any error occurs during encryption.

    Example:
        >>> encrypted_vector = encrypt_vector([1.2, 3.4, 5.6], context)
        This encrypts the vector [1.2, 3.4, 5.6] using the provided CKKS context.

    Note:
        Ensure the TenSEAL library (as ts) and the logging library are imported.
        This function uses the CKKS encryption scheme from TenSEAL for vector encryption.
    """
    try:
        # Check if vector is a list and not empty
        if not isinstance(vector, list) or len(vector) == 0:
            raise ValueError("Vector must be a non-empty list.")

        # Log the beginning of the encryption process
        logging.info("Starting encryption process...")

        encrypted_vector = ts.ckks_vector(context, vector)
        # Log the end of the encryption process
        logging.info("Encryption process completed successfully.")

        return encrypted_vector
    except Exception as e:
        logging.error(f"An error occurred during the encryption process: {e}")
        raise

def decrypt_vector(encrypted_vector):
    """Decrypt a vector encrypted with the CKKS scheme."""
    logging.info("Starting decryption process...")
    decrypted_vector = encrypted_vector.decrypt()
    logging.info("Decryption process completed successfully.")
    return decrypted_vector


def recommend_items(user_index, similar_users_indices, data, n=10):
    """
    Recommends items to a user based on the ratings of similar users.

    This function identifies items that similar users have rated highly (rating of 4 or more)
    and which the target user hasn't rated yet. The items are then ranked based on their
    popularity among the similar users.

    Parameters:
        user_index (int): Index of the target user for recommendations.
        similar_users_indices (list[int]): Indices of users considered similar to the target user.
        data (pd.DataFrame): Data frame containing users' ratings. Expected columns: 'User_ID', 'Item_ID', 'Rating'.
        n (int, optional): Number of recommendations desired. Defaults to 10.

    Returns:
        list[int]: IDs of the top n recommended items.

    Raises:
        ValueError: If data doesn't contain the expected columns or if ratings are outside the range 1-5.
        IndexError: If user_index or any index in similar_users_indices is out of bounds.

    Example:
        >>> recommend_items(0, [1, 2, 3], data)
        This will determine the top 10 items to recommend to the user at index 0 based on
        the ratings from users at indices 1, 2, and 3.

    Note:
        Ensure that the pandas library (as pd) is imported before using this function.
    """
    if not {'User_ID', 'Item_ID', 'Rating'}.issubset(data.columns):
        raise ValueError("The data DataFrame must contain 'User_ID', 'Item_ID', and 'Rating' columns.")

    unique_users = data['User_ID'].unique()

    if user_index < 0 or user_index >= len(unique_users):
        raise IndexError("Invalid user_index. It's out of the range of unique users in the data.")

    for idx in similar_users_indices:
        if idx < 0 or idx >= len(unique_users):
            raise IndexError(
                f"Invalid index {idx} in similar_users_indices. It's out of the range of unique users in the data.")

    if not (isinstance(n, int) and n > 0):
        raise ValueError("n should be a positive integer.")

    ratings = data['Rating']
    if not all(1 <= rating <= 5 for rating in ratings):
        raise ValueError("All ratings should be between 1 and 5.")

    # Continue with the original function logic
    user_id = unique_users[user_index]
    user_items = set(data[data['User_ID'] == user_id]['Item_ID'])
    recommendation_counts = {}

    for similar_user_index in similar_users_indices:
        similar_user_id = unique_users[similar_user_index]
        similar_user_highly_rated_items = set(
            data[(data['User_ID'] == similar_user_id) & (data['Rating'] >= 4)]['Item_ID']
        )
        recommended_items = similar_user_highly_rated_items - user_items
        for item in recommended_items:
            recommendation_counts[item] = recommendation_counts.get(item, 0) + 1

    recommended_items_sorted = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
    top_recommended_items = [item for item, count in recommended_items_sorted[:n]]

    return top_recommended_items


def generate_synthetic_data(participant_count: int, items_per_participant: int) -> None:
    """
    Generates synthetic data for a given number of participants with specified items per participant.

    This function creates synthetic data for a defined number of participants, where each participant
    has a certain number of items. Each item consists of a random user ID and a random rating.
    The generated data is saved into individual Excel files for each participant and a
    combined Excel file for all participants.

    Parameters:
        participant_count (int): Number of participants for which data should be generated.
        items_per_participant (int): Number of items per participant.

    Raises:
        ValueError: If participant_count or items_per_participant is not a positive integer.

    Example:
        >>> generate_synthetic_data(10, 5)
        This will generate data for 10 participants, each with 5 items. It will create 10
        individual Excel files (e.g., 'participant_1_data.xlsx', 'participant_2_data.xlsx')
        and one combined Excel file named 'combined_participant_data.xlsx'.

    Note:
        Ensure that pandas (pd) and random libraries are imported before using this function.
    """

    if not (isinstance(participant_count, int) and participant_count > 0):
        raise ValueError("participant_count should be a positive integer.")

    if not (isinstance(items_per_participant, int) and items_per_participant > 0):
        raise ValueError("items_per_participant should be a positive integer.")

    data = []

    for participant_id in range(1, participant_count + 1):
        participant_data = []
        for item_id in range(1, items_per_participant + 1):
            user_id = random.randint(1000, 9999)  # Generate random user ID
            rating = random.randint(1, 5)  # Generate random rating

            participant_data.append({'User_ID': user_id, 'Item_ID': item_id, 'Rating': rating})

        data.extend(participant_data)

        # Handle potential file write errors
        try:
            # Create a separate Excel file for each participant
            participant_df = pd.DataFrame(participant_data)
            excel_filename = f'participant_{participant_id}_data.xlsx'
            participant_df.to_excel(excel_filename, index=False)
            print(f'Synthetic data for Participant {participant_id} saved to {excel_filename}')
        except Exception as e:
            print(f"Error saving synthetic data for Participant {participant_id}. Error: {e}")

    # Handle potential file write errors for the combined file
    try:
        # Create a combined Excel file for all participants
        combined_df = pd.DataFrame(data)
        combined_excel_filename = 'combined_participant_data.xlsx'
        combined_df.to_excel(combined_excel_filename, index=False)
        print(f'Combined synthetic data for all participants saved to {combined_excel_filename}')
    except Exception as e:
        print(f"Error saving combined synthetic data. Error: {e}")


def detect_anomalies(data):
    """
    Perform machine learning data analysis and anomaly detection on synthetic user-item interaction data.

    This function conducts feature engineering to extract user behavior patterns, visualizes key features,
    and detects anomalies in the data using the Isolation Forest algorithm. Additionally, it checks for
    data consistency based on predefined rules.

    Parameters:
       data (pd.DataFrame): The synthetic data containing 'User_ID', 'Item_ID', and 'Rating' columns.

    Raises:
       ValueError: If anomalies are detected or data consistency rules are violated.
    """
    # Feature Engineering
    user_item_count = data.groupby('User_ID')['Item_ID'].count()
    user_avg_rating = data.groupby('User_ID')['Rating'].mean()
    user_rating_std = data.groupby('User_ID')['Rating'].std()
    user_max_rating = data.groupby('User_ID')['Rating'].max()
    user_min_rating = data.groupby('User_ID')['Rating'].min()
    user_unique_items = data.groupby('User_ID')['Item_ID'].nunique()
    user_rating_skew = data.groupby('User_ID')['Rating'].skew()

    user_features = pd.concat([user_item_count, user_avg_rating, user_rating_std,
                               user_max_rating, user_min_rating, user_unique_items,
                               user_rating_skew], axis=1)
    user_features.columns = ['num_items_rated', 'avg_rating', 'rating_std',
                             'max_rating', 'min_rating', 'unique_items_rated', 'rating_skew']
    user_features.fillna(0, inplace=True)

    min_valid_rating = 1
    max_valid_rating = 5
    min_user_items = 0
    min_item_ratings = 1

    invalid_ratings = data[(data['Rating'] < min_valid_rating) | (data['Rating'] > max_valid_rating)]
    users_with_few_items = user_item_count[user_item_count < min_user_items]
    items_with_few_ratings = data['Item_ID'].value_counts()[data['Item_ID'].value_counts() < min_item_ratings]

    if not invalid_ratings.empty:
        raise ValueError("Invalid ratings detected!")
    if not users_with_few_items.empty:
        raise ValueError("Users with too few rated items detected!")
    if not items_with_few_ratings.empty:
        raise ValueError("Items with too few ratings detected!")

    # Anomaly Detection using Isolation Forest
    clf = IsolationForest(contamination=0.1)
    anomalies = clf.fit_predict(user_features)
    user_features['anomaly'] = anomalies
    anomalous_users = user_features[user_features['anomaly'] == -1]

    # Raising an error if anomalies are detected
    if not anomalous_users.empty:
        raise ValueError("Anomalies detected in the synthetic data!")

    try:
        result = "No anomalies detected."
    except ValueError as e:
        result = str(e)

    clf = IsolationForest(contamination=0.1)
    anomalies = clf.fit_predict(user_features)
    user_features['anomaly'] = anomalies
    anomalous_users = user_features[user_features['anomaly'] == -1]

    # Save the trained Isolation Forest model
    model_path = "isolation_forest_model.pkl"
    joblib.dump(clf, model_path)

    # Log anomalies instead of raising an error
    if not anomalous_users.empty:
        with open("/mnt/data/anomaly_log.txt", "a") as log_file:
            log_file.write(f"Anomalies detected in the synthetic data at {os.path.basename(data)}:\n")
            log_file.write(str(anomalous_users))
            log_file.write("\n\n")

    return "Anomaly detection complete. Check the log for any detected anomalies."


def factorize_and_recommend(data, user_id, num_items=5):
    """
    Factorizes the user-item matrix using SVD and returns the top-rated items for a given user.

    Parameters:
      data: DataFrame containing User_ID, Item_ID, and Rating columns.
      user_id: ID of the user.
      num_items: Number of top-rated items to return.

    Returns:
      Top-rated items for the user.
    """
    # Ensure data contains required columns
    required_columns = {'User_ID', 'Item_ID', 'Rating'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data should contain the following columns: {required_columns}")

    # Ensure user_id exists in the data
    if user_id not in data['User_ID'].values:
        raise ValueError(f"User_ID {user_id} not found in the data.")

    # Transform the data into a user-item matrix
    user_item_matrix = data.pivot(index='User_ID', columns='Item_ID', values='Rating').fillna(0)

    # Ensure the data is not empty
    if user_item_matrix.empty:
        raise ValueError("User-Item matrix is empty. Please provide valid data.")

    # Ensure num_items is not greater than total items in the dataset
    if num_items > user_item_matrix.shape[1]:
        raise ValueError(f"Requested number of items ({num_items}) is greater than available items ({user_item_matrix.shape[1]}).")


    # Perform Singular Value Decomposition
    U, sigma, Vt = np.linalg.svd(user_item_matrix, full_matrices=False)

    def top_rated_items_internal(user_id, U, sigma, Vt, num_items=5):
        """
        Internal function to return the top-rated items for a given user.
        """
        # Convert user_id to user index
        user_index = user_item_matrix.index.get_loc(user_id)

        # Convert sigma to a diagonal matrix
        sigma_matrix = np.diag(sigma)

        # Calculate the predicted ratings for the user
        predicted_ratings = np.dot(np.dot(U[user_index, :], sigma_matrix), Vt)

        # Get the top-rated items
        top_items_indices = np.argsort(predicted_ratings)[::-1][:num_items]
        top_items = [user_item_matrix.columns[i] for i in top_items_indices]

        return top_items

    return top_rated_items_internal(user_id, U, sigma, Vt, num_items)


def decrypt_in_secure_environment(encrypted_data):
    """Decrypt data in a secure environment using TenSEAL."""
    decrypted_data = [encrypted_row.decrypt() for encrypted_row in encrypted_data]
    return decrypted_data

def perform_matrix_factorization(decrypted_data):
    """Perform matrix factorization on the decrypted data using SVD."""
    U, sigma, Vt = np.linalg.svd(decrypted_data, full_matrices=False)
    return U, sigma, Vt

def generate_recommendations(user_index, decrypted_data, U, sigma, Vt, num_items=5):
    """Generate recommendations for a user based on the factorized matrices."""
    # Convert sigma to a diagonal matrix
    sigma_matrix = np.diag(sigma)

    # Calculate the predicted ratings for the user
    predicted_ratings = np.dot(np.dot(U[user_index, :], sigma_matrix), Vt)

    # Recommend the top-rated items
    top_rated_item_indices = np.argsort(predicted_ratings)[::-1][:num_items]

    reencrypt_data(decrypted_data)

    return top_rated_item_indices

def reencrypt_data(data):
    """Re-encrypt data in the secure environment before exiting using TenSEAL."""
    encrypted_data = [ts.ckks_vector(context, row) for row in data]
    return encrypted_data

def recommend_for_user(user_index, encrypted_data, num_items=5):
    """Get recommendations for a user."""
    decrypted_data = decrypt_in_secure_environment(encrypted_data)
    U, sigma, Vt = perform_matrix_factorization(decrypted_data)
    recommended_items = generate_recommendations(user_index, decrypted_data, U, sigma, Vt, num_items)

    return recommended_items


def generate_user_item_matrix(data):
    """Generate a user-item matrix from the provided data."""

    user_item_matrix = data.pivot(index='User_ID', columns='Item_ID', values='Rating').fillna(0)

    return user_item_matrix