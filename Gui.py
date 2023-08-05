import random
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pandas as pd
import tenseal as ts
from main import create_user_vectors, generate_synthetic_data, encrypt_vector, calculate_similarities, \
    find_similar_users, recommend_items


lab_report = '''
Title: Privacy-Preserving Item Recommendation System Using Homomorphic Encryption

Abstract: This report describes the implementation of a privacy-preserving recommendation system. 
The system uses homomorphic encryption to ensure user privacy while processing item recommendation based on user-item ratings. 
The CKKS encryption scheme and the TenSEAL library are used for the homomorphic encryption. 
The system also includes functionalities for loading user-item rating data, generating synthetic data, calculating cosine
 similarities, and recommending items based on similar users' ratings.

Introduction: Recommendation systems are widely used in various online platforms to suggest items to users based on their 
preferences. However, these systems often require access to sensitive user data, which raises privacy concerns. 
Homomorphic encryption, a cryptographic method that allows computations on encrypted data, can be a solution to this problem.
This report discusses the implementation of a privacy-preserving recommendation system using homomorphic encryption.

Methods: The Python script uses the pandas library for data manipulation, numpy for numerical computations, 
and the TenSEAL library for homomorphic encryption. User-item rating data is loaded from an Excel file or generated synthetically.
The data is then processed to create a matrix representing user ratings for different items. 
The user vectors are encrypted using the CKKS encryption scheme provided by TenSEAL. 
The system calculates the cosine similarities between the encrypted user vectors and finds the most similar users to a 
given user. The items rated highly by these similar users, which have not been rated by the given user, are then 
recommended to the given user.

Results: Depends on the users generated synthetic data, it will return the recommended item for the user.

Discussion: The implemented system can provide item recommendations while preserving user privacy by using homomorphic encryption.
However, the system's performance and the quality of recommendations depend on the underlying user-item rating data. 
Furthermore, the use of homomorphic encryption can increase the computational complexity of the system.

Conclusion: The implementation of a privacy-preserving recommendation system using homomorphic encryption demonstrates 
the potential of homomorphic encryption in protecting user privacy in recommendation systems. Future work can focus on 
optimizing the system's performance and evaluating the quality of recommendations.
'''
def show_lab_report():
    messagebox.showinfo("Lab Report", lab_report)


def generate_synthetic_data_gui():
    # Get the number of participants and items per participant from the input fields
    try:
        participant_count = int(participants_input.get())
        items_per_participant = int(items_input.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid integers for the number of participants and "
                                            "items per participant.")
        return

    try:
        generate_synthetic_data(participant_count, items_per_participant)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    messagebox.showinfo("Success", "Synthetic data generated successfully.")

def load_data_and_recommend_items():
    # Open a file dialog for the user to select the Excel file
    file_path = filedialog.askopenfilename()

    if not file_path:  # Check that a file was selected
        messagebox.showerror("File Error", "Please select a valid Excel file.")
        return

    try:
        # Load the data
        data = pd.read_excel(file_path)

        # Create user vectors
        user_vectors = create_user_vectors(data)

        # Create a TenSEAL context for the encryption
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = 2 ** 40
        context.generate_galois_keys()

        # Encrypt the user vectors and calculate their norms
        encrypted_vectors = []
        norms = []
        for vector in user_vectors:
            encrypted_vectors.append(encrypt_vector(vector, context))
            norms.append(np.linalg.norm(vector))

        # Choose a user to make recommendations for
        user_id = random.randint(0, len(user_vectors)-1)

        # Calculate the similarities
        similarities = calculate_similarities(user_id, encrypted_vectors, norms)

        # Find the most similar users
        similar_users = find_similar_users(user_id, similarities)

        # Recommend items
        recommendations = recommend_items(user_id, similar_users, data)

        if len(recommendations) > 0:
            messagebox.showinfo("Recommendations", "Recommended items: " + ", ".join(map(str, recommendations)))
        else:
            messagebox.showinfo("Recommendations", "No recommended items found.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

# Create a window
window = tk.Tk()

# Create labels and input fields for the number of participants and items per participant
participants_label = tk.Label(window, text="Enter number of users:")
participants_input = tk.Entry(window)
items_label = tk.Label(window, text="Enter number of items:")
items_input = tk.Entry(window)

show_report_button = tk.Button(window, text="Show Lab Report", command=show_lab_report)

# Create a button for generating synthetic data
generate_button = tk.Button(window, text="Generate Synthetic Data", command=generate_synthetic_data_gui)

# Create a button for loading data and recommending items
recommend_button = tk.Button(window, text="Load Data and Recommend Items", command=load_data_and_recommend_items)

# Add the labels, input fields, and buttons to the window
participants_label.pack()
participants_input.pack()
items_label.pack()
items_input.pack()
generate_button.pack()
recommend_button.pack()
show_report_button.pack()

# Start the event loop
window.mainloop()