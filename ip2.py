import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox, font

# Loading the data of ipl 2008-2024 on the bases of the winners of every match
def load_data(deliveries_path, matches_path):
    """Load deliveries and matches datasets"""
    delv = pd.read_csv(deliveries_path)
    match = pd.read_csv(matches_path)
    return delv, match

# Combine the data of two dataframes
def merge_data(delv, match):
    """Merge deliveries and matches on match_id"""
    if 'match_id' in delv.columns and 'id' in match.columns:
        merged = match.merge(delv, left_on='id', right_on='match_id', how='inner')
        return merged
    else:
        raise ValueError("Required columns are missing in the DataFrames.")

# Preparing the data for the output by the result
def prepare_data(match):
    """Prepare data for model training by encoding team names and defining the target variable"""
    match['result'] = (match['team1'] == match['winner']).astype(int)
    enc = LabelEncoder()
    
    # Fit encoder only on unique team names to ensure correct mapping
    teams = pd.concat([match['team1'], match['team2']]).unique()
    enc.fit(teams)
    
    match['team1'] = enc.transform(match['team1'])
    match['team2'] = enc.transform(match['team2'])
    
    x = match[['team1', 'team2']]
    y = match['result']
    return x, y, enc, teams  # return teams for dropdown

# Training the 80% data and testing on 20% remaining data
def train_model(x, y):
    """Train a logistic regression model"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

# Predict the model
def predict_match(model, enc, team1, team2):
    """Predict the match outcome between two teams"""
    try:
        if team1 not in enc.classes_ or team2 not in enc.classes_:
            return None, None
        
        e_team1 = enc.transform([team1])[0]
        e_team2 = enc.transform([team2])[0]
        
        user_data = [[e_team1, e_team2]]
        probabilities = model.predict_proba(user_data)
        
        win_prob = probabilities[0][1] * 100
        lose_prob = probabilities[0][0] * 100
        
        return win_prob, lose_prob
    except ValueError:
        return None, None

# Creating a messagebox for the given input
def custom_messagebox(title, message):
    """Create a custom messagebox with larger text and adjustable size."""
    custom_box = tk.Toplevel(root)
    custom_box.title(title)
    custom_box.geometry("700x500")
    
    label_font = font.Font(family="Helvetica", size=14, weight="bold")
    label = tk.Label(custom_box, text=message, font=label_font)
    label.pack(pady=20)
    
    ok_button = tk.Button(custom_box, text="OK", command=custom_box.destroy, font=("Helvetica", 12))
    ok_button.pack(pady=10)
    
    custom_box.transient(root)
    custom_box.grab_set()
    root.wait_window(custom_box)

# Update the on_predict function to use dropdown values
def on_predict():
    """Handle predict button click in GUI"""
    team1 = selected_team1.get()
    team2 = selected_team2.get()
    
    if team1 == "Select Team 1" or team2 == "Select Team 2":
        custom_messagebox("Error", "Please select both teams.")
        return

    win_prob, lose_prob = predict_match(model, enc, team1, team2)
    
    if win_prob is not None and lose_prob is not None:
        custom_messagebox("Prediction Result", 
                          f"{team1} wins with probability: {win_prob:.2f}%\n"
                          f"{team2} wins with probability: {lose_prob:.2f}%")
    else:
        custom_messagebox("Error", "Invalid team names. Please select valid teams.")

# Load data and train model
try:
    delv, match = load_data("deliveries.csv", "matches.csv")
    merged_delv = merge_data(delv, match)
    x, y, enc, teams = prepare_data(merged_delv)  # Get team names for dropdown
    model = train_model(x, y)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load data or train model: {str(e)}")
    raise

# Create the main window
root = tk.Tk()
root.title("Match Prediction")
root.geometry("700x500")

# Define fonts
entry_font = font.Font(family="Helvetica", size=15)
label_font = font.Font(family="Helvetica", size=22, weight="bold")
button_font = font.Font(family="Helvetica", size=20)

# Define tk.StringVar for each dropdown to hold the selected team
selected_team1 = tk.StringVar(root)
selected_team2 = tk.StringVar(root)

# Set default values for dropdowns
selected_team1.set("Select Team 1")
selected_team2.set("Select Team 2")

# Create labels and dropdowns for team selection
tk.Label(root, text="Select your team:", font=label_font).pack(pady=5)
team1_dropdown = tk.OptionMenu(root, selected_team1, *teams)  # Use team names for dropdown
team1_dropdown.config(font=entry_font)
team1_dropdown.pack(pady=5)

tk.Label(root, text="Select 2nd team:", font=label_font).pack(pady=5)
team2_dropdown = tk.OptionMenu(root, selected_team2, *teams)
team2_dropdown.config(font=entry_font)
team2_dropdown.pack(pady=5)

# Create a predict button
predict_button = tk.Button(root, text="Predict", command=on_predict, font=button_font)
predict_button.pack(pady=20)

# Run the application
root.mainloop()
