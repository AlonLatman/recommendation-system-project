import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import webbrowser
from main import (
    encrypt_data,
    generate_shares,
    apply_differential_privacy,
    train_model,
    recommend_items
)

LAB_REPORT = """
Lab Report
----------

This is the lab report for the data encryption project. The goal of this project is to demonstrate how to perform data
encryption and differential privacy to protect sensitive user-item ratings data while preserving utility for recommendation.

The process involves the following steps:

1. Data Loading: Load user-item ratings data from an Excel file.
2. Encryption: Encrypt the data using a secure encryption algorithm with a randomly generated key.
3. Secret Sharing: Generate shares of the encrypted data to distribute trust across multiple parties.
4. Differential Privacy: Apply differential privacy to the shares to protect individual user ratings.
5. Matrix Factorization: Train a matrix factorization model on the differentially private data.
6. Item Recommendation: Recommend items to a specific user based on the trained model.

Implementation Details:
------------------------

1. Data Loading:
   - The user provides an Excel file containing user-item ratings data.
   - The data is loaded into a pandas DataFrame for further processing.

2. Encryption:
   - The data is converted into a numerical matrix format.
   - The numerical matrix is encrypted using a secure encryption algorithm.
   - The encryption process is performed in a way that ensures the data privacy is maintained.

3. Secret Sharing:
   - The encrypted data is divided into multiple shares using a secret sharing scheme.
   - The number of shares required to reconstruct the original data is specified by the threshold.

4. Differential Privacy:
   - Differential privacy is applied to the shares to add noise and protect individual user ratings.
   - Laplace noise is commonly used in differential privacy mechanisms.

5. Matrix Factorization:
   - The differentially private shares are combined to reconstruct the encrypted matrix.
   - Matrix factorization is applied to the reconstructed matrix to learn the latent features of users and items.

6. Item Recommendation:
   - The matrix factorization model is used to predict user-item ratings.
   - Based on the predicted ratings, items are recommended to a specific user.

Example:
---------

Suppose we have the following user-item ratings data:

| User_ID | Item_ID | Rating |
|---------|---------|--------|
| 1       | A       | 4.5    |
| 1       | B       | 3.0    |
| 2       | A       | 2.0    |
| 2       | C       | 5.0    |
| 3       | B       | 1.5    |
| 3       | C       | 4.0    |

The user-item matrix looks like this:

|         | A    | B    | C    |
|---------|------|------|------|
| User_ID |      |      |      |
| 1       | 4.5  | 3.0  | 0.0  |
| 2       | 2.0  | 0.0  | 5.0  |
| 3       | 0.0  | 1.5  | 4.0  |

After encryption and secret sharing, the differentially private matrix may look like:

|         | A                    | B                    | C                    |
|---------|----------------------|----------------------|----------------------|
| User_ID |                      |                      |                      |
| Share 1 | 4.509                | 3.007                | 0.0                  |
| Share 2 | 2.021                | 0.0                  | 4.999                |
| Share 3 | 0.0                  | 1.486                | 3.991                |

Using the matrix factorization model, we can predict the user-item ratings:

| User_ID | Item_ID | Predicted Rating |
|---------|---------|------------------|
| 1       | A       | 4.512            |
| 1       | B       | 2.994            |
| 1       | C       | 3.005            |
| 2       | A       | 2.008            |
| 2       | B       | 3.011            |
| 2       | C       | 4.999            |
| 3       | A       | 2.999            |
| 3       | B       | 1.487            |
| 3       | C       | 4.000            |

Based on the predicted ratings, we can recommend items to specific users. For example, the recommended items for User 1 would be [A, C], for User 2 would be [C], and for User 3 would be [A, C].

Instructions:
1. Click the 'Browse' button to select an Excel file containing user-item ratings data.
2. Click the 'Encrypt Data' button to perform data encryption, differential privacy, and item recommendation.
3. The recommended items for user ID 5 will be displayed in a message box.

"""


class DataEncryptionApp:
    def __init__(self, root):
        self.root = root
        self.file_path = None

        self.file_path_entry = tk.Entry(root, width=50)
        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.encrypt_button = tk.Button(root, text="Encrypt Data", command=self.encrypt_and_recommend)

        self.file_path_entry.pack(pady=10)
        self.browse_button.pack(pady=5)
        self.encrypt_button.pack(pady=10)

    def browse_file(self):
        self.file_path = filedialog.askopenfilename()

        if self.file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, self.file_path)

    def show_lab_report(self):
        # Create a new window to show the lab report content
        lab_report_window = tk.Toplevel(self.root)
        lab_report_window.title("Lab Report")
        lab_report_window.geometry("800x600")

        # Create a text widget to display the lab report content
        lab_report_text = tk.Text(lab_report_window, wrap=tk.WORD)
        lab_report_text.insert(tk.END, LAB_REPORT)
        lab_report_text.pack(expand=True, fill=tk.BOTH)

    def encrypt_and_recommend(self):
        if self.file_path:
            try:
                user_id = 5
                df = pd.read_excel(self.file_path)
                user_item_grouped = df.groupby(['user_id', 'item_id'], as_index=False).mean()
                user_item_matrix = user_item_grouped.pivot(index='user_id', columns='item_id', values='rating')
                user_item_matrix.fillna(0, inplace=True)
                data = user_item_matrix.to_numpy(dtype=np.float32)

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

                # Show the recommended items in a message box.
                self.show_message("Recommended Items", f"Recommended items for user {user_id}: {recommended_items}")

            except Exception as e:
                self.show_message("Error", f"Error loading file: {e}")
        else:
            self.show_message("Error", "Please select a data file.")

    def show_message(self, title, message):
        tk.messagebox.showinfo(title, message)


class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recommender System GUI")

        # Add the "Links and explanations" text line
        self.label = tk.Label(root, text="Links & Explanations", font=("Helvetica", 14, "bold"), width=25)
        self.label.pack()

        # Create the Differential privacy button
        self.privacy_button = tk.Button(root, text="Differential privacy", command=self.open_wikipedia_page, bg="green")
        self.privacy_button.pack()

    def open_wikipedia_page(self):
        # Function to open the Wikipedia page explaining Differential privacy
        webbrowser.open("https://en.wikipedia.org/wiki/Differential_privacy")


class GUIApp2:
    def __init__(self, root):
        self.root = root
        self.root.title("Recommender System GUI")
        # Create the Differential privacy button
        self.privacy_button = tk.Button(root, text="Matrix_factorization", command=self.open_wikipedia_page, bg="green")
        self.privacy_button.pack()

    def open_wikipedia_page(self):
        # Function to open the Wikipedia page explaining Differential privacy
        webbrowser.open("https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)")


class GUIApp3:
    def __init__(self, root):
        self.root = root
        self.root.title("Recommender System GUI")
        # Create the Differential privacy button
        self.privacy_button = tk.Button(root, text="Blum Blum Shub", command=self.open_wikipedia_page, bg="green")
        self.privacy_button.pack()

    def open_wikipedia_page(self):
        # Function to open the Wikipedia page explaining Differential privacy
        webbrowser.open("https://en.wikipedia.org/wiki/Blum_Blum_Shub")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Data Encryption App")
    # boot the main GUI window
    app = DataEncryptionApp(root)
    # Add the "Lab Report" button
    lab_report_button = tk.Button(root, text="Lab Report", command=app.show_lab_report)
    lab_report_button.pack(pady=5)
    # Add 3 buttons with links to wikipedia
    app = GUIApp(root)
    app = GUIApp2(root)
    app = GUIApp3(root)

    root.mainloop()