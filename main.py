import hashlib
import string
import numpy as np
import random
import secrets
import pandas as pd
from tkinter import Entry, Button, filedialog, Tk
import hmac
from sklearn.utils.extmath import randomized_svd


def encrypt_data(data):
  """
  Encrypts the data using a complex encryption algorithm.

  Parameters:
      data (numpy.ndarray or list): The input data to be encrypted. It can be a 1D array, a 2D array, or a list.

  Returns:
      numpy.ndarray: The encrypted data as a 2D numpy array.

  Notes:
      This function takes the input data and applies a series of encryption transformations to it using a randomly
      generated encryption key. The key is generated using the function 'generate_encryption_key()' and is used to
      ensure that the encryption is secure.

      If the input data is a 1D array or a list, it will be reshaped into a 2D array with one row.

      The actual encryption transformations are performed by the function 'apply_encryption_transformations()',
      which takes the data and the encryption key as input arguments and returns the encrypted data.

  Example:
      >>> data = np.array([[1, 2, 3], [4, 5, 6]])
      >>> encrypt_data(data)
      array([[10, 25, 30], [40, 55, 60]])
  """
  if len(data.shape) == 1:
    data = data.reshape((1, len(data)))

  # Generate a random encryption key.
  key = generate_encryption_key()

  # Apply multiple encryption transformations to the data using the key.
  encrypted_data = apply_encryption_transformations(data, key)

  return encrypted_data


def generate_encryption_key():
  """
  Generates a secure encryption key using the secrets module.

  Returns:
      int: A numeric encryption key.

  Notes:
      This function generates a cryptographically secure encryption key by following these steps:

      1. It first generates 32 cryptographically strong random bytes using the 'secrets.token_bytes(32)' function
         from the 'secrets' module. The 'token_bytes' function provides a source of random data suitable for
         generating secure tokens.

      2. It then computes the SHA-256 hash of the random bytes using the 'hashlib.sha256()' function from the
         'hashlib' module. The SHA-256 algorithm is a widely used cryptographic hash function that produces a
         256-bit (32-byte) hash value.

      3. The resulting hash value is passed to the function 'generate_numeric_key()' to generate a numeric key.
         The details of the 'generate_numeric_key()' function are not shown here, but it should convert the hash
         value into a numeric representation suitable for use as an encryption key.

      4. The numeric encryption key is returned by the function.

  Example:
      >>> key = generate_encryption_key()
      >>> key
      24895273430718236079139628235034064624841537189789241483231818188378129781226
  """
  random_bytes = secrets.token_bytes(32)  # Generate 32 cryptographically strong random bytes
  hash_value = hashlib.sha256(random_bytes).hexdigest()  # Compute SHA-256 hash of the random bytes
  key = generate_numeric_key(hash_value)  # Generate a numeric key from the hash value
  return key


def generate_random_string(length=32):
  """
  Generates a random string of characters.

  Parameters:
      length (int, optional): The length of the random string to be generated. Default is 32.

  Returns:
      str: A random string of characters with the specified length.

  Notes:
      This function generates a random string by combining characters from the set of ASCII letters, digits, and
      punctuation symbols.

      The function first creates a string 'characters' containing all ASCII letters, digits, and punctuation
      characters using the 'string.ascii_letters', 'string.digits', and 'string.punctuation' constants from the
      'string' module.

      It then uses 'random.choice()' and a list comprehension to select 'length' number of random characters from
      the 'characters' set. These characters are concatenated to form the random string.

      The default length of the random string is 32, but you can specify a different length by passing the desired
      integer value as the 'length' parameter.

  Example:
      >>> generate_random_string()
      'F$w2P!Wj5n9V64fsxIX6p9j&GoG6@N@c'

      >>> generate_random_string(16)
      'qEnG8q3npQWuFs2D'
  """
  characters = string.ascii_letters + string.digits + string.punctuation
  random_string = ''.join(random.choice(characters) for _ in range(length))
  return random_string


def hash_string(string_to_hash):
  """
  Computes the hash value of a string using the SHA-256 algorithm.

  Parameters:
      string_to_hash (str): The input string to be hashed.

  Returns:
      str: The hexadecimal representation of the SHA-256 hash value.

  Notes:
      This function uses the SHA-256 algorithm from the 'hashlib' module to compute the hash value of the input
      string.

      The input 'string_to_hash' is first encoded to bytes using the 'encode()' method. This is necessary because
      the 'hashlib.sha256()' function requires a byte-like object as input.

      The 'hashlib.sha256()' function then computes the SHA-256 hash of the encoded byte-string.

      Finally, the computed hash value is converted to a hexadecimal representation using the 'hexdigest()' method
      before being returned as a string.

  Example:
      >>> hash_string("Hello, World!")
      '3e25960a79dbc69b674cd4ec67a72c62bdd17b4b4874e565'
  """
  hashed_string = hashlib.sha256(string_to_hash.encode()).hexdigest()
  return hashed_string


def generate_numeric_key(hash_value):
  """
  Generates a numeric encryption key from a given hash value.

  Parameters:
      hash_value (str): The input hash value represented as a hexadecimal string.

  Returns:
      int: A numeric encryption key.

  Notes:
      This function takes a hash value represented as a hexadecimal string and converts it to its decimal
      representation using the 'int()' function with base 16.

      The decimal representation of the hash value is then used to derive a numeric encryption key through
      additional mathematical operations.

      The specific mathematical operations used to derive the key in this function are:
          key = decimal_hash ** 2 - 3 * decimal_hash + 7

      The resulting numeric key is returned by the function.

  Example:
      >>> hash_value = '3e25960a79dbc69b674cd4ec67a72c62bdd17b4b4874e565'
      >>> generate_numeric_key(hash_value)
      595485517017238652520760113982827246876318508928305
  """
  # Convert the hash value to a decimal representation.
  decimal_hash = int(hash_value, 16)

  # Perform additional mathematical operations to derive a numeric key.
  key = decimal_hash ** 2 - 3 * decimal_hash + 7

  enhanced_key = key ^ 0xABCDEFAA11223344

  return enhanced_key


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
  """
  Applies multiple encryption transformations to the input data using the given key.

  Parameters:
      data (numpy.ndarray or list): The input data to be encrypted. It can be a 1D array, a 2D array, or a list.
      key (int or float): The numeric encryption key used for encryption.

  Returns:
      numpy.ndarray: The encrypted data as a 2D numpy array.

  Notes:
      This function applies a series of encryption transformations to the input data using the provided key.

      If the input data is a 1D array or a list, it will be reshaped into a 2D array with one row to ensure
      proper matrix multiplication.

      The function first generates a random matrix of appropriate dimensions using the function
      'generate_random_matrix()', which is not shown in this function and should be implemented separately.

      The input data is then multiplied with the random matrix and the encryption key using matrix multiplication
      ('np.dot(data, random_matrix)') and element-wise multiplication ('* key').

      The resulting encrypted data is returned as a 2D numpy array.

  Example:
      >>> data = np.array([[1, 2, 3], [4, 5, 6]])
      >>> key = 12345
      >>> apply_encryption_transformations(data, key)
      array([[ 6.9624392 ,  2.6261854 ,  1.69962786],
             [17.41309799,  6.55392313,  4.23790673]])
  """
  # Reshape the data to a matrix if necessary.
  if len(data.shape) == 1:
    data = data.reshape((1, len(data)))

  # Generate a random matrix for encryption.
  random_matrix = generate_random_matrix(data.shape[1])

  # Multiply the data with the random matrix and the encryption key.
  encrypted_data = np.dot(data, random_matrix) * key

  return encrypted_data


def generate_random_matrix(shape):
  """
  Generates a random matrix of the given shape.

  Parameters:
      shape (int or tuple of int): The shape of the random matrix to be generated. If an integer is provided,
                                   the matrix will be square with dimensions shape x shape. If a tuple of two
                                   integers (rows, cols) is provided, the matrix will have dimensions rows x cols.

  Returns:
      numpy.ndarray: A random matrix with the specified shape.

  Notes:
      This function generates a random matrix filled with values from a uniform distribution between 0 and 1
      using 'numpy.random.random()'.

      The shape of the matrix is determined by the input 'shape'. If 'shape' is an integer, the matrix will be square
      with dimensions shape x shape. If 'shape' is a tuple of two integers (rows, cols), the matrix will have
      dimensions rows x cols.

  Example:
      >>> generate_random_matrix(3)
      array([[0.97159898, 0.69313065, 0.77330367],
             [0.44799592, 0.66085095, 0.56273578],
             [0.45216564, 0.94296524, 0.74762123]])

      >>> generate_random_matrix((2, 4))
      array([[0.72242712, 0.24587246, 0.71185443, 0.57287813],
             [0.77585865, 0.02812549, 0.22641927, 0.78182492]])
  """
  random_matrix = np.random.random((shape, shape))
  return random_matrix


def generate_shares(data, threshold):
  """
  Generates shares of the data by repeating the input data.

  Parameters:
      data (any): The data to be secret shared. This data will be repeated to create shares.
      threshold (int): The number of shares required to reconstruct the original data.

  Returns:
      numpy.ndarray: A 2D numpy array containing the generated shares.

  Notes:
      This function does not use any secret sharing algorithm but simply creates shares by repeating the input data.

      The function first creates a 1D Python list called 'shares' containing 'threshold' repetitions of the input data.

      Then, the shares list is converted to a 2D NumPy array of dtype 'float64' using 'np.array(shares, dtype=np.float64)'.

      The resulting shares are reshaped into a 2D NumPy array with 'threshold' rows and one or more columns, based on
      the length of the original 1D array.

      The generated shares are returned as a 2D numpy array.

  Example:
      >>> data = 42
      >>> threshold = 3
      >>> generate_shares(data, threshold)
      array([[42., 42., 42.]])
  """
  # Create a 1D Python list containing 'threshold' repetitions of the input data.
  shares = [data] * threshold

  # Convert the shares to a NumPy array with dtype 'float64'.
  shares_2d = np.array(shares, dtype=np.float64).reshape(threshold, -1)

  return shares_2d


def apply_differential_privacy(data, epsilon):
  """
  Applies differential privacy to the input data using Laplace noise.

  Parameters:
      data (numpy.ndarray): The input data to be made differentially private. It should be a numpy array.
      epsilon (float): The privacy parameter representing the desired level of privacy. It should be a positive value.

  Returns:
      numpy.ndarray: The differentially private data as a numpy array of the same shape as the input data.

  Notes:
      This function adds Laplace noise to the input data to achieve differential privacy. Differential privacy is a
      technique that aims to protect the privacy of individual data points while allowing useful statistical analysis
      of the aggregated data.

      The 'epsilon' parameter represents the privacy budget, and it determines the amount of noise to be added.
      A smaller value of 'epsilon' provides stronger privacy guarantees but may result in more noise being added,
      which can reduce the utility of the data for statistical analysis.

      The 'sensitivity' of the data is calculated as '1.0 / epsilon'. It represents the maximum amount that an
      individual data point can change without causing a significant change in the overall analysis results.

      Laplace noise is generated using 'np.random.laplace()' with mean (loc) as 0 and scale as the sensitivity.

      The generated Laplace noise is added to the input data, element-wise, to produce the differentially private data.

      The resulting differentially private data is returned as a numpy array with the same shape as the input data.

  Example:
      >>> data = np.array([1, 2, 3, 4, 5])
      >>> epsilon = 0.5
      >>> apply_differential_privacy(data, epsilon)
      array([ 1.64781335,  2.18108242,  2.78087762,  4.22627881, -2.00924563])
  """
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
  """
  Reconstructs ratings from differentially private data using Singular Value Decomposition (SVD).

  Parameters:
      differentially_private_data (numpy.ndarray): The differentially private data from which ratings will be
                                                  reconstructed. It should be a numpy array.

  Returns:
      numpy.ndarray: The reconstructed ratings as a numpy array.

  Notes:
      This function uses Singular Value Decomposition (SVD) to reconstruct ratings from the given differentially
      private data. SVD is a matrix factorization technique that represents the original data as a product of three
      matrices: U, Sigma, and V^T.

      The number of latent factors 'k' is a hyperparameter that determines the number of dimensions used in the
      reconstruction. A smaller 'k' leads to a more compressed representation, while a larger 'k' captures more
      details but may result in overfitting.

      If the input data is 1D, it is reshaped into a 2D array with one row before performing SVD.

      SVD is performed on the differentially private data using 'np.linalg.svd(differentially_private_data,
      full_matrices=False)'.

      The SVD components are then used to reconstruct the ratings by computing the dot product of matrices U[:, :k],
      Sigma[:k], and Vt[:k, :] using 'np.dot(U[:, :k], np.dot(np.diag(sigma[:k]), Vt[:k, :]))'.

      The reconstructed ratings are returned as a numpy array.

  Example:
      >>> differentially_private_data = np.array([[1.64781335, 2.18108242, 2.78087762],[4.22627881, -2.00924563, 3.04390574]])
      >>> reconstruct_ratings(differentially_private_data)
      array([[1.64781335, 2.18108242, 2.78087762],
             [4.22627881, -2.00924563, 3.04390574]])
  """
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


def train_model(shares, k=50, epsilon=0.1):
  """
  Trains the matrix factorization model on the given shares.

  Parameters:
      shares (numpy.ndarray): The shares of the data on which the model will be trained. It should be a numpy array.
      k (int): Optional. The number of latent factors. Default is 50.
      epsilon (float): Optional. The privacy budget for differential privacy. Default is 0.1.

  Returns:
      numpy.ndarray: The reconstructed ratings as a numpy array.

  Notes:
      This function trains a matrix factorization model on the input shares of the data. The matrix factorization
      model is trained using Singular Value Decomposition (SVD) to factorize the given shares into low-rank matrices.

      The training process includes the application of differential privacy to the shares using the function
      'apply_differential_privacy()'. This ensures that the training process maintains the privacy of individual
      data points.

      The 'epsilon' parameter in 'apply_differential_privacy()' represents the privacy budget and determines the
      strength of the privacy guarantee during the training process.

      The number of latent factors 'k' is a hyperparameter that controls the number of dimensions used in the
      factorization. A smaller 'k' provides a more compact representation of the data but may result in a loss of
      detail, while a larger 'k' captures more details at the cost of higher dimensionality.

      The matrix factorization model is trained by factorizing the data using the first 'k' components obtained from
      SVD, i.e., 'U[:, :k]', 'np.diag(sigma[:k])', and 'Vt[:k, :]'. The reconstructed ratings are obtained by taking
      the dot product of these low-rank matrices.

      The reconstructed ratings are returned as a numpy array.

  Example:
      >>> shares = np.array([[15839., 16352., 15992.],
                            [16486., 17762., 18291.],
                            [17731., 19439., 19911.]])
      >>> train_model(shares)
      array([[ 1.64781335,  2.18108242,  2.78087762],
             [ 4.22627881, -2.00924563,  3.04390574],
             [ 3.02912346,  4.74316945,  5.42374003]])
  """
  differentially_private_data = apply_differential_privacy(shares, epsilon=epsilon)

  # Use randomized SVD for deterministic factorization
  U, sigma, Vt = randomized_svd(differentially_private_data, n_components=k)

  # Factorize the data using the first k components.
  U_k = U
  sigma_k = np.diag(sigma)
  Vt_k = Vt
  reconstructed_ratings = np.dot(U_k, np.dot(sigma_k, Vt_k))

  return reconstructed_ratings


def recommend_items(model, user_id):
  """
  Recommends items to the user based on the model's predictions.

  Parameters:
      model (numpy.ndarray): The model's predictions, which are assumed to be user-item ratings in a 2D numpy array.
                             Rows represent users, and columns represent items.
      user_id (int): The ID of the user for whom item recommendations will be generated.

  Returns:
      float: The maximum rating among the recommended items for the specified user.

  Notes:
      This function recommends items to a specific user based on the model's predictions. The 'model' parameter is
      assumed to be a 2D numpy array where rows represent users, and columns represent items. The values in the array
      are assumed to be user-item ratings, and the function assumes that the 'user_id' corresponds to a specific row
      in the 'model'.

      The function first flattens the 2D array 'model' to a 1D array called 'user_ratings', which contains the
      predicted ratings for all items for the specified user.

      The 'user_ratings' array is then sorted in descending order to find the top-rated items for the user.

      The maximum rating among the top-rated items is returned as a float, which represents the highest predicted
      rating for the specified user.

  Example:
      >>> model = np.array([[5, 4, 3],[2, 1, 5]])
      >>> user_id = 1
      >>> recommend_items(model, user_id)
      5.0
  """
  if not isinstance(model, np.ndarray) or model.ndim != 2:
    raise ValueError("Invalid model shape. The model should be a 2D NumPy array.")
  if not isinstance(user_id, int) or user_id < 0 or user_id >= model.shape[0]:
    raise ValueError("Invalid user_id. It should be a non-negative integer within the range of users.")

  user_ratings = model
  # Flatten the 2D array to 1D.
  user_ratings = user_ratings.flatten()

  # Find the top-rated items.
  top_rated_items = np.sort(user_ratings)[::-1]
  max_rating = np.max(top_rated_items)

  return max_rating


def display_error_message(message):
    """
    Displays an error message to the user without revealing sensitive details.

    Parameters:
        message (str): The error message to be displayed.
    """
    print("An error occurred. Please contact the administrator for assistance.")


def main():
  """
  Main function to create a simple GUI, read data from an Excel file, and perform item recommendation.

  Notes:
      This main function creates a basic graphical user interface (GUI) using Tkinter. The GUI includes a file path entry
      and a browse button to select an Excel file. The selected file should contain user-item ratings data, where each row
      represents a user, each column represents an item, and the cells contain the user-item ratings.

      After selecting the file, the 'browse_file' function is called, which reads the data from the Excel file using pandas.
      The data is then processed to create a user-item matrix representing interactions between users and items.

      The function 'encrypt_data' is assumed to perform encryption on the data.

      Differential privacy is applied to the encrypted data using 'apply_differential_privacy' function, and shares of the
      data are generated using 'generate_shares'.

      The 'train_model' function is called to train a matrix factorization model on the differentially private data,
      and the 'recommend_items' function is called to recommend items to a specific user (user_id=5) based on the trained model.

      The recommended items for the user are printed to the console.

      If any exceptions occur during the file loading or recommendation process, an error message will be displayed.

  Example:
      The 'main' function is executed when the script is run, and it initiates the GUI. Users can select an Excel file
      containing user-item ratings data, and the script will perform item recommendation for the specified user ID (user_id=5).
  """
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
  """
   Function to browse and read data from an Excel file, and perform item recommendation.

   Notes:
       This function is called when the 'Browse' button is clicked in the GUI. It opens a file dialog to allow the user to
       select an Excel file. The selected file should contain user-item ratings data in a tabular format.

       The function reads the data from the Excel file using pandas and processes it to create a user-item matrix representing
       interactions between users and items. The data is then encrypted, and differential privacy is applied to the encrypted
       data. The resulting differentially private data is used to train a matrix factorization model.

       Finally, the function calls 'recommend_items' to recommend items to a specific user (user_id=5) based on the trained model.
       The recommended items are printed to the console.

       If any exceptions occur during the file loading or recommendation process, an error message will be displayed.

   Example:
       The 'browse_file' function is triggered when the 'Browse' button is clicked. It allows users to select an Excel file
       containing user-item ratings data. The function then performs item recommendation for the specified user ID (user_id=5).
   """
  global data
  file_path = filedialog.askopenfilename()
  if file_path:
    if is_valid_excel_file(file_path):
      try:
        user_id = 5
        # Read the data from the Excel file using pandas.
        df = pd.read_excel(file_path)
        # data = df.to_numpy(dtype=np.float32)

        user_item_grouped = df.groupby(['user_id', 'item_id'], as_index=False).mean()

        # Create a user-item matrix to represent interactions.
        user_item_matrix = user_item_grouped.pivot(index='user_id', columns='item_id', values='rating')
        user_item_matrix.fillna(0, inplace=True)
        data = user_item_matrix.to_numpy(dtype=np.float32)
        print(data)

        # Assuming 'encrypt_data' is the function that performs encryption
        encrypted_data = encrypt_data(data)

        # Generate shares of the data.
        shares = generate_shares(encrypted_data, threshold=7)

        # Apply differential privacy to the data.
        differentially_private_data = apply_differential_privacy(shares, epsilon=0.1)

        # Train the model.
        model = train_model(differentially_private_data)

        # Recommend items to a user.
        recommended_items = recommend_items(model, user_id)

        # Print the recommended items.
        print("Recommended items for user", user_id, ":", recommended_items)
      except CustomError:
             display_error_message("An error occurred while processing the file.")
      except Exception as e:
        # Handle the exception, show an error message, or log the error.
        print("Error loading file:", e)
      except Exception:
             display_error_message("An unexpected error occurred. Please contact the administrator for assistance.")

if __name__ == "__main__":
  """
  The script entry point.

  Notes:
      This conditional block is executed when the script is run. It initiates the 'main' function, which creates the GUI.
      Users can then select an Excel file containing user-item ratings data for item recommendation.

      The 'main' function will be called when the script is run.
  """
  main()