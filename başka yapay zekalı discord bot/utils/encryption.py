import os
from cryptography.fernet import Fernet

def encrypt_data(data: str) -> str:
    key = os.getenv('ENCRYPTION_KEY').encode()
    cipher_suite = Fernet(key)
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    key = os.getenv('ENCRYPTION_KEY').encode()
    cipher_suite = Fernet(key)
    return cipher_suite.decrypt(encrypted_data.encode()).decode()