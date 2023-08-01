# recommendation-system-project
This repository contains Python code for a simple recommender system based on matrix factorization. The code uses Singular Value Decomposition (SVD) to factorize user-item interaction data and provide item recommendations to users.

* Table of Contents
* Introduction
* Dependencies
* Usage
* Functions
* Example
Introduction
The code in this repository provides a basic matrix factorization-based recommender system. It demonstrates how to perform matrix factorization using Singular Value Decomposition (SVD) to extract latent features from user-item interaction data. The latent features are then used to make item recommendations to users.

# Dependencies
The code depends on the following Python libraries:

numpy: For numerical operations and matrix manipulations.
pandas: For data loading and manipulation.
scikit-learn: For SVD-based matrix factorization.
secretsharing: For generating shares of data.
tkinter: For creating a simple GUI.
You can install the required libraries using pip:
"pip install numpy pandas scikit-learn secretsharing"

# Usage
Clone this repository to your local machine.
Install the required dependencies as mentioned in the Dependencies section.
The main code is in the recommender_system.py file. You can run the code from the terminal using:
"python recommender_system.py"
The script will open a simple graphical user interface (GUI) using Tkinter. You can browse and select an Excel file containing user-item ratings data.
The code will perform matrix factorization and provide item recommendations for a specific user based on the trained model.

# Functions
The main code includes the following functions:

apply_differential_privacy: Applies differential privacy to the data using Laplace noise.
encrypt_data: Encrypts the data using a complex encryption algorithm.
generate_shares: Generates shares of the data by repeating the input data.
generate_numeric_key: Generates a numeric encryption key from a given hash value.
generate_random_matrix: Generates a random matrix of the given shape.
generate_random_string: Generates a random string of characters.
generate_encryption_key: Generates a secure encryption key using the secrets module.
hash_string: Computes the hash value of a string using the SHA-256 algorithm.
train_model: Trains the matrix factorization model on the given shares.
reconstruct_ratings: Reconstructs ratings from differentially private data using SVD.
recommend_items: Recommends items to the user based on the model's predictions.

# Example
The following example demonstrates how to use the code to create a recommender system for user-item ratings data:
<pre>
```python
import pandas as pd
import numpy as np
from recommender_system import apply_differential_privacy, encrypt_data, generate_shares, train_model, recommend_items
df = pd.read_excel("user_item_ratings.xlsx")
user_item_grouped = df.groupby(['user_id', 'item_id'], as_index=False).mean()
user_item_matrix = user_item_grouped.pivot(index='user_id', columns='item_id', values='rating')
user_item_matrix.fillna(0, inplace=True)
data = user_item_matrix.to_numpy(dtype=np.float32)
encrypted_data = encrypt_data(data)
shares = generate_shares(encrypted_data, threshold=7)
differentially_private_data = apply_differential_privacy(shares, epsilon=0.1)
model = train_model(differentially_private_data)
user_id = 5
recommended_items = recommend_items(model, user_id)
print("Recommended items for user", user_id, ":", recommended_items)
</pre>

The code reads user-item ratings data from an Excel file, performs matrix factorization with differential privacy, and recommends items for a specific user based on the trained model.
