import os
import random
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
import pandas as pd
import tenseal as ts
from main import create_user_vectors, generate_synthetic_data, encrypt_vector, calculate_similarities, \
    find_similar_users, recommend_items
import webbrowser


lab_report = '''
Title: Privacy-Preserving Item Recommendation System Using Homomorphic Encryption

Abstract: 
This report elucidates the design and deployment of a recommendation system that accentuates user privacy through the 
application of homomorphic encryption. 
Utilizing the CKKS encryption scheme coupled with the TenSEAL library, this system processes item recommendations 
predicated on user-item interactions while maintaining data confidentiality. 
Key functionalities encompass user-item data loading, synthetic data generation, computation of cosine similarities, 
and item recommendations based on analogous user preferences.

Introduction: 
Recommendation systems have become an integral part of online platforms, enhancing user experiences by personalizing item 
suggestions in alignment with individual preferences. 
Despite their ubiquity, a significant drawback is their reliance on sensitive user data, leading to legitimate privacy 
apprehensions. As a potential resolution, homomorphic encryption – a form of cryptography enabling computations on 
ciphered data – offers a promising avenue. 
This document delves into the intricacies of developing a recommendation system fortified with homomorphic encryption techniques.

Methods: 
The recommendation system was developed with a focus on user privacy and accuracy in suggesting items. 
The methodology can be broken down into the following steps:
Data Acquisition:
User-item interactions, a cornerstone for recommendations, can be sourced in two ways:
* Direct ingestion from an Excel datasheet.
* Generation of synthetic data, mimicking real-world user interactions, providing a sandbox environment for testing and validation.
Data Processing:
Once acquired, this data undergoes transformation into a matrix format, where rows represent individual users and columns 
depict items. Each cell in this matrix illustrates a user's rating or interaction level with a particular item.
Encryption:
To ensure user privacy, each user's interaction vector is encrypted. The CKKS encryption scheme, provided by the TenSEAL library, 
is employed for this purpose. This encryption allows computations on data without ever decrypting it, ensuring user data remains confidential throughout the processing.
Similarity Computation:
The crux of the recommendation logic lies in discerning users with similar tastes or preferences. This is achieved by 
computing cosine similarities between encrypted vectors. 
The outcome is a similarity score that indicates how alike two users are in terms of their item interactions.
Item Recommendations:
Based on similarity scores, the system identifies users who share preferences with the target user. 
Recommendations are then generated by collating items that these similar users have rated highly but the target user 
hasn't interacted with.

Results:
During the testing phase, the system demonstrated a balance between preserving user privacy and providing accurate recommendations. 
The synthetic data tests exhibited a recommendation accuracy of approximately 100% for 10 items and above for 5 items the accuracy was 70%. 
Moreover, the computational overhead introduced by encryption was found to be manageable, with processing times increasing 
by approximately 40% compared to non-encrypted data.
Please take note that even with 20 items because the data is synthetic you could receive only 2 recommendations depends on the
rating given when the data is created.

Discussion: 
The implemented system can provide item recommendations while preserving user privacy by using homomorphic encryption.
However, the system's performance and the quality of recommendations depend on the underlying user-item rating data. 
Furthermore, the use of homomorphic encryption can increase the computational complexity of the system.

Conclusion: 
The integration of homomorphic encryption into the recommendation system successfully addressed privacy concerns associated 
with user-item interaction data. 
While there's an undeniable computational overhead due to encryption, the benefits in terms of user trust and data security 
make it a viable solution for platforms that prioritize user privacy. Future work may delve into optimizing processing times 
and exploring other encryption schemes to further enhance system performance.
'''
def show_lab_report():
    """
    Opens the lab report in a window
    """
    messagebox.showinfo("Lab Report", lab_report)

def open_webpage():
    """
    Opens the Wikipedia page for "CKKS" in the default web browser.
    """
    webbrowser.open('https://he.wikipedia.org/wiki/CKKS')

def open_webpage2():
    """
    Opens the Wikipedia page for "Recommender System" in the default web browser.
    """
    webbrowser.open('https://en.wikipedia.org/wiki/Recommender_system')


def open_webpage3():
    """
    Opens the Wikipedia page for "Cosine similarity" in the default web browser.
    """
    webbrowser.open('https://en.wikipedia.org/wiki/Cosine_similarity')


def generate_synthetic_data_gui():
    """
    Generates synthetic data through a GUI by fetching the number of participants and items per participant from input fields.

    Description:
        The function is designed to be used in a GUI environment.
        It fetches the number of participants and items per participant from the specified input fields.
        If the input values are not valid integers, an error message is displayed to the user.
        After validating the inputs, the function calls the generate_synthetic_data method with the input values to generate the synthetic data.
        If any exception occurs during this process, an error message is displayed to the user.
        On successful generation of synthetic data, an informational message is shown to the user.

    Assumptions:
        The participants_input and items_input objects have a get() method that fetches the values from the respective input fields in the GUI.
        The messagebox module is used to display error or informational messages to the user.
        The generate_synthetic_data function is defined elsewhere and takes two integer parameters:
        the number of participants and items per participant.
    """
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
    """
    Loads user-item data from an Excel file, processes the data, and makes item recommendations for a randomly selected user.

    The function performs the following steps:
    1. Opens a file dialog for the user to select the Excel file.
    2. Reads and loads the data from the Excel file.
    3. Creates user vectors based on the loaded data.
    4. Initializes a TenSEAL context for encryption.
    5. Encrypts the user vectors and calculates their norms.
    6. Randomly selects a user.
    7. Calculates similarities between the selected user and other users.
    8. Identifies the most similar users based on the calculated similarities.
    9. Makes item recommendations for the selected user based on the items liked by similar users.
    10. Displays the recommended items to the user through a message box.

    If any error occurs during the process, an error message is displayed to the user.

    Assumptions:
    - The Excel file contains the necessary columns for user-item interactions.
    - Required modules (`pandas`, `numpy`, `TenSEAL`, etc.) have been imported.
    - Supporting functions (`create_user_vectors`, `encrypt_vector`, `calculate_similarities`, `find_similar_users`, and `recommend_items`) are defined elsewhere and are compatible with the data structure.

    Raises:
    - `Exception`: If there's an error during data processing or recommendation generation.
    """
    # Open a file dialog for the user to select the Excel file
    file_path = "combined_participant_data.xlsx"

    if not os.path.exists(file_path):  # Check that the file exists
        messagebox.showerror("File Error", "Data file not found. Please ensure combined_participant_data.xlsx is in the project folder.")
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
window.title('Privacy-Preserving Item Recommendation System')


# Create a frame for the synthetic data generation controls
synthetic_data_frame = ttk.Frame(window, padding='3 3 12 12')
synthetic_data_frame.grid(column=0, row=0, sticky=(tk.W, tk.E))

# Create labels and input fields for the number of participants and items per participant
participants_label = ttk.Label(synthetic_data_frame, text='Number of users:')
participants_input = ttk.Entry(synthetic_data_frame)
items_label = ttk.Label(synthetic_data_frame, text='Number of items per user:')
items_input = ttk.Entry(synthetic_data_frame)

# Create a button for generating synthetic data
generate_button = ttk.Button(synthetic_data_frame, text='Generate Synthetic Data', command=generate_synthetic_data_gui)

# Position the controls in the grid
participants_label.grid(column=0, row=0, sticky=tk.W)
participants_input.grid(column=1, row=0, sticky=(tk.W, tk.E))
items_label.grid(column=0, row=1, sticky=tk.W)
items_input.grid(column=1, row=1, sticky=(tk.W, tk.E))
generate_button.grid(column=0, row=2, columnspan=2)

# Create a frame for the recommendation controls
recommendation_frame = ttk.Frame(window, padding='3 3 12 12')
recommendation_frame.grid(column=0, row=1, sticky=(tk.W, tk.E))

# Create a button for loading data and recommending items
recommend_button = ttk.Button(recommendation_frame, text='Recommend Items', command=load_data_and_recommend_items)

# Position the control in the grid
recommend_button.grid(column=0, row=0)

# Create a frame for the lab report control
lab_report_frame = ttk.Frame(window, padding='3 3 12 12')
lab_report_frame.grid(column=0, row=2, sticky=(tk.W, tk.E))

# Create a button for showing the lab report
show_report_button = ttk.Button(lab_report_frame, text='Show Lab Report', command=show_lab_report)

# Position the control in the grid
show_report_button.grid(column=0, row=0)

# Create a new frame for the button
button_frame = ttk.Frame(window, padding='3 3 12 12')
button_frame.grid(column=0, row=3, sticky=(tk.W, tk.E))

# Create a button for opening the Wikipedia page
open_webpage_button = ttk.Button(button_frame, text='Open CKKS Wikipedia Page', command=open_webpage)

# Position the button in the grid
open_webpage_button.grid(column=0, row=0)

button_frame = ttk.Frame(window, padding='3 3 12 12')
button_frame.grid(column=0, row=4, sticky=(tk.W, tk.E))

# Create a button for opening the Wikipedia page
open_webpage_button = ttk.Button(button_frame, text='Open Recommendation System Wikipedia Page', command=open_webpage2)

# Position the button in the grid
open_webpage_button.grid(column=0, row=0)

button_frame = ttk.Frame(window, padding='3 3 12 12')
button_frame.grid(column=0, row=5, sticky=(tk.W, tk.E))

# Create a button for opening the Wikipedia page
open_webpage_button = ttk.Button(button_frame, text='Open Cosine similarity Wikipedia Page', command=open_webpage3)

# Position the button in the grid
open_webpage_button.grid(column=0, row=0)

# Start the event loop
window.mainloop()