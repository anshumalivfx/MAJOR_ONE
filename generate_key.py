import pickle
from pathlib import Path

import streamlit_authenticator as stauth


names = ["Lakshita Gupta", "Bhavey Mittal", "Anshumali Karna"]
usernames = ["lakshita", "bhavey", "anshumali"]
password = ["admins", "admins", "admins"]


hashed_passwords = stauth.Hasher(password).generate()

file_path = Path(__file__).parent / "hashed_passwords.pkl"

with open(file_path, "wb") as f:
    pickle.dump(hashed_passwords, f)