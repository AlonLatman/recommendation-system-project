import hashlib
import pickle
import string
import numpy as np
import random
import secrets
import pandas as pd
from tkinter import Entry, Button, filedialog, Tk
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from secrets import token_bytes
import os


def encrypt_data(data, participant_id):
  # Convert participant's ID to bytes
  participant_id_bytes = str(participant_id).encode('utf-8')

  # Generate random salt and IV
  salt = os.urandom(32)
  initial_vector = os.urandom(16)

  # Generate encryption key using PBKDF2HMAC
  kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    iterations=100000,
    salt=salt,
    length=32,
    backend=default_backend()
  )

  key = kdf.derive(participant_id_bytes)

  # Initialize AES cipher with the derived key and a random IV
  cipher = Cipher(algorithms.AES(key), modes.CFB(initial_vector), backend=default_backend())
  encryptor = cipher.encryptor()

  # Encrypt data
  padding_data = sym_padding.PKCS7(128).padder()
  padded_data = padding_data.update(data) + padding_data.finalize()
  encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

  return initial_vector + salt + encrypted_data


def decrypt_data(encrypted_data, participant_id):
  # Convert participant's ID to bytes
  participant_id_bytes = str(participant_id).encode('utf-8')

  # The first 16 bytes of the encrypted data are the IV
  initial_vector = encrypted_data[:16]

  # The next 32 bytes of the encrypted data are the salt
  salt = encrypted_data[16:48]

  # Generate decryption key using PBKDF2HMAC
  kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    iterations=100000,
    salt=salt,
    length=32,
    backend=default_backend()
  )

  key = kdf.derive(participant_id_bytes)

  # Initialize AES cipher with the derived key and IV using CFB mode
  cipher = Cipher(algorithms.AES(key), modes.CFB(initial_vector), backend=default_backend())
  decryptor = cipher.decryptor()

  # Decrypt and remove padding
  decrypted_data = decryptor.update(encrypted_data[48:]) + decryptor.finalize()
  unpadder = sym_padding.PKCS7(128).unpadder()
  unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()

  return unpadded_data


def generate_synthetic_data(participant_count, items_per_participant):
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


def generate_encryption_key():
  random_bytes = secrets.token_bytes(32)  # Generate 32 cryptographically strong random bytes
  hash_value = hashlib.sha256(random_bytes).hexdigest()  # Compute SHA-256 hash of the random bytes
  key = generate_numeric_key(hash_value)  # Generate a numeric key from the hash value
  return key


def generate_random_string(length=32):
  characters = string.ascii_letters + string.digits + string.punctuation
  random_string = ''.join(random.choice(characters) for _ in range(length))
  return random_string


def hash_string(string_to_hash):
  hashed_string = hashlib.sha256(string_to_hash.encode()).hexdigest()
  return hashed_string


def generate_numeric_key(hash_value):
  if not isinstance(hash_value, str):
    raise TypeError("The secret_key should be a string.")

  # Hash the secret_key using SHA-256
  hashed_key = hashlib.sha256(hash_value.encode()).hexdigest()

  # Convert the hashed_key to an integer
  key = int(hashed_key, 16)

  # Truncate or pad the numeric_key to the specified length
  key %= 10 ** 16

  return key


class CustomError(Exception):
  """Custom exception class for better error handling and reporting."""

  def __init__(self, message, error_code=None):
    super().__init__(message)
    self.error_code = error_code


def is_valid_excel_file(file_path):
  """
  Validates if the given file is a valid Excel file.

  Parameters:
      file_path (str): The path to the Excel file to be validated.

  Returns:
      bool: True if the file is a valid Excel file, False otherwise.
  """
  try:
    if not file_path.endswith(".xlsx"):
      raise CustomError("Invalid file format. Please provide an Excel file.", error_code=1)

    return True
  except CustomError as custom_error:
    raise custom_error
  except Exception as e:
    raise CustomError("An error occurred while validating the input file.", error_code=2)


try:
  file_path = "path_to_your_excel_file.xlsx"
  if is_valid_excel_file(file_path):
    print("Excel file validation boot up successful.")
  else:
    print("Excel file validation failed.")
except CustomError as custom_error:
  if custom_error.error_code == 1:
    print("Error:", custom_error)
    print("Please provide an Excel file.")
  elif custom_error.error_code == 2:
    print("Error:", custom_error)
    print("An unexpected error occurred during validation.")
  else:
    print("An unknown error occurred:", custom_error)
except Exception as e:
  print("An unexpected error occurred:", e)


def apply_encryption_transformations(data, key):
  # Reshape the data to a matrix if necessary.
  if len(data.shape) == 1:
    data = data.reshape((1, len(data)))

  # Generate a random matrix for encryption.
  random_matrix = generate_random_matrix(data.shape[1])

  # Multiply the data with the random matrix and the encryption key.
  encrypted_data = np.dot(data, random_matrix) * key

  return encrypted_data


def generate_random_matrix(shape):
  random_matrix = np.random.random((shape, shape))
  return random_matrix


def generate_shares(encrypted_data, num_participants):
    # Generate random secret keys for each participant
    secret_keys = [token_bytes(32) for _ in range(num_participants)]

    # Initialize a list to hold shares for each participant
    shares_by_participant = [[] for _ in range(num_participants)]


    # Split the encrypted data into shares
    for byte in encrypted_data:
      byte_shares = secret_to_shares(byte, num_participants)
      for participant, share in enumerate(byte_shares):
        shares_by_participant[participant].append(share)

    # Secure aggregation: Combine shares using the Secure Sum protocol
    max_num_shares = max(len(participant_shares) for participant_shares in shares_by_participant)

    # Pad the shares of participants with fewer shares
    for participant_shares in shares_by_participant:
      while len(participant_shares) < max_num_shares:
        participant_shares.append(0)

    # Perform the aggregation
    shares_array = np.array(shares_by_participant)
    aggregated_shares = np.sum(shares_array, axis=0)

    # Serialize secret keys, aggregated shares, and other participant data
    serialized_data_list = []
    for participant in range(num_participants):
      participant_data = {
        'secret_key': secret_keys[participant],
        'aggregated_shares': aggregated_shares,
        'other_participant_data': serialize_other_participant_data(participant, shares_by_participant)
      }
      serialized_data = serialize_data(participant_data)
      serialized_data_list.append(serialized_data)

    return serialized_data_list


def secret_to_shares(secret, num_shares):
  coefficients = [secret] + [random.randint(0, 255) for _ in range(num_shares - 1)]
  shares = [(i, evaluate_polynomial(coefficients, i)) for i in range(1, num_shares + 1)]
  return shares


def evaluate_polynomial(coefficients, x):
  result = 0
  power = 1
  for coefficient in coefficients:
    result += coefficient * power
    power *= x
  return result


def serialize_data(data):
  serialization = pickle.dumps(data)
  return serialization


def serialize_other_participant_data(participant, shares_by_participant):
  serialized_other_participant_data = []
  for p, shares in enumerate(shares_by_participant):
    if p != participant:
      serialized_other_participant_data.append(serialize_data({'participant': p, 'shares': shares}))
  return serialized_other_participant_data


def apply_differential_privacy(data, epsilon):
  if not isinstance(data, np.ndarray):
    raise TypeError("Invalid data type. The data should be a NumPy array.")

  sensitivity = 1.0 / epsilon
  shape = data.shape
  noise = np.random.laplace(loc=0, scale=sensitivity, size=shape)

  # Rescale the noise to meet the privacy requirements more accurately
  max_noise = np.max(np.abs(noise))
  privacy_threshold = sensitivity
  scale_factor = min(1.0, privacy_threshold / max_noise)
  noise *= scale_factor

  differentially_private_data = data + noise

  # Check if L1 sensitivity is preserved
  max_sensitivity_difference = np.max(np.abs(differentially_private_data - data))
  assert max_sensitivity_difference <= privacy_threshold, "Privacy requirements not met"

  return differentially_private_data


def reconstruct_ratings(differentially_private_data):
  # Reshape the data to 2D if it is 1D.
  if len(differentially_private_data.shape) == 1:
    differentially_private_data = differentially_private_data.reshape(1, -1)

  # Perform SVD on the differentially private data.
  U, sigma, Vt = np.linalg.svd(differentially_private_data, full_matrices=False)

  # Reconstruct the ratings using the SVD components.
  k = 50  # Number of latent factors (a hyperparameter).
  reconstructed_data = np.dot(U[:, :k], np.dot(np.diag(sigma[:k]), Vt[:k, :]))

  # If the data was originally 1D, reshape it back to 1D after SVD computation.
  if len(differentially_private_data.shape) == 1:
    reconstructed_data = reconstructed_data.ravel()

  return reconstructed_data


def train_model(participant_data, num_epochs, learning_rate):
    participant_data = decrypt_data(participant_data,participant_id=2)
    print(participant_data)
    # Initialize model parameters
    num_features = len(participant_data[0]['input_features'])
    model_weights = np.random.rand(num_features)

    for epoch in range(num_epochs):
      aggregated_gradients = np.zeros(num_features)

      for participant_data in participant_data:
        encrypted_gradients = compute_encrypted_gradients(participant_data, model_weights)
        aggregated_gradients += encrypted_gradients

      # Securely aggregate the encrypted gradients
      securely_aggregated_gradients = secure_aggregation(aggregated_gradients, len(participant_data))

      # Update model weights using the aggregated gradients
      model_weights -= learning_rate * securely_aggregated_gradients

    return model_weights


def compute_encrypted_gradients(participant_data, model_weights):
  # Decrypt the participant's shares using their secret key
  decrypted_shares = decrypt_shares(participant_data['aggregated_shares'], participant_data['secret_key'])

  # Compute gradients on the decrypted data
  gradients = compute_gradients(decrypted_shares, model_weights)

  # Encrypt the gradients using the participant's public key
  encrypted_gradients = encrypt_gradients(gradients, participant_data['public_key'])

  return encrypted_gradients


def decrypt_shares(aggregated_shares, secret_key):
  # Decryption process using the secret key
  decrypted_shares = [share ^ secret_key for share in aggregated_shares]
  return decrypted_shares


def compute_gradients(decrypted_shares, model_weights):
  # Example: Compute gradients using decrypted shares and model weights
  gradients = np.array([share * feature for share, feature in zip(decrypted_shares, model_weights)])
  return gradients


def encrypt_gradients(gradients, public_key):
  # Encryption process using the participant's public key
  encrypted_gradients = [gradient ^ public_key for gradient in gradients]
  return encrypted_gradients


def secure_aggregation(aggregated_gradients, num_participants):
  # Secure aggregation process (e.g., using Secure Sum)
  securely_aggregated_gradients = aggregated_gradients / num_participants
  return securely_aggregated_gradients


def recommend_items(trained_model_weights, participant_data_list):
    recommended_items = []

    for participant_data in participant_data_list:
      encrypted_model_weights = encrypt_model_weights(trained_model_weights, participant_data['public_key'])
      encrypted_scores = compute_encrypted_scores(encrypted_model_weights, participant_data['encrypted_features'])
      decrypted_scores = decrypt_scores(encrypted_scores, participant_data['secret_key'])

      recommended_item = np.argmax(decrypted_scores)  # Recommend the item with the highest score
      recommended_items.append(recommended_item)

    return recommended_items


def encrypt_model_weights(model_weights, public_key):
  encrypted_model_weights = [weight ^ public_key for weight in model_weights]
  return encrypted_model_weights


def compute_encrypted_scores(encrypted_model_weights, encrypted_features):
  encrypted_scores = [weight ^ feature for weight, feature in zip(encrypted_model_weights, encrypted_features)]
  return encrypted_scores


def decrypt_scores(encrypted_scores, secret_key):
  decrypted_scores = [score ^ secret_key for score in encrypted_scores]
  return decrypted_scores


def display_error_message(message):
  """
  Displays an error message to the user without revealing sensitive details.

  Parameters:
      message (str): The error message to be displayed.
  """
  print("An error occurred. Please contact the administrator for assistance.")


def main():
  # Create the GUI.
  root = Tk()

  # Create the file path entry.
  file_path_entry = Entry(root)

  # Create the browse button.
  browse_button = Button(root, text="Browse", command=lambda: browse_file())

  # Layout the GUI.
  file_path_entry.pack()
  browse_button.pack()

  # Start the GUI.
  root.mainloop()


def browse_file():

 generate_synthetic_data(participant_count=3, items_per_participant=5)
global data
file_path = filedialog.askopenfilename()
if file_path:
  try:
    if is_valid_excel_file(file_path):
      df = pd.read_excel(file_path)

      # Process the data and create a user-item matrix
      user_item_grouped = df.groupby(['User_ID', 'Item_ID'], as_index=False).mean()
      user_item_matrix = user_item_grouped.pivot(index='User_ID', columns='Item_ID', values='Rating')
      user_item_matrix.fillna(0, inplace=True)
      data = user_item_matrix.to_numpy(dtype=np.float32)

      # Encrypt the data
      encrypted_data = encrypt_data(data, participant_id=2)

      # Generate shares of the encrypted data
      participant_data_list = generate_shares(encrypted_data, num_participants=3)

      # participant_data_list = np.array(participant_data_list)

      # Train the model using the shares
      trained_model_weights = train_model(participant_data_list, num_epochs=20, learning_rate=0.1)

      # Recommend items to a user
      recommended_items = recommend_items(trained_model_weights, participant_data_list)

      # Print the recommended items
      print("Recommended items for user 5:", recommended_items)

  except CustomError as custom_error:
    display_error_message("An error occurred: " + str(custom_error))
  except Exception as e:
    display_error_message("An unexpected error occurred: " + str(e))

if __name__ == "__main__":
  """
  The script entry point.

  Notes:
      This conditional block is executed when the script is run. It initiates the 'main' function, which creates the GUI.
      Users can then select an Excel file containing user-item ratings data for item recommendation.

      The 'main' function will be called when the script is run.
  """
  main()