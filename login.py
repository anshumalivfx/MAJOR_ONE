import sqlite3
import hashlib
import uuid

# Function to create a connection to the SQLite database
def create_connection():
    connection = sqlite3.connect('users.db')
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    connection.commit()
    return connection

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to sign up a new user
def signup(name, username, password):
    connection = create_connection()
    cursor = connection.cursor()

    user_id = str(uuid.uuid4())  # Generate a unique UUID for the user

    hashed_password = hash_password(password)

    cursor.execute('INSERT INTO users (name, username, password) VALUES (?, ?, ?, ?)', (name, username, hashed_password))
    connection.commit()

    print(f"User {username} signed up successfully!")


# Function to log in a user
def login(username, password):
    connection = create_connection()
    cursor = connection.cursor()

    hashed_password = hash_password(password)

    cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hashed_password))
    user = cursor.fetchone()

    if user:
        print(f"Welcome, {username}!")
        return {
            "id": user[0],
            "name": user[1],
            "username": user[2]
        }
    else:
        print("Invalid username or password.")
        return None
