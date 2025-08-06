import nltk
import os
os.makedirs("/root/nltk_data", exist_ok=True)
nltk.data.path.append("/root/nltk_data")
try:
    nltk.download("averaged_perceptron_tagger_eng", download_dir="/root/nltk_data")
    print("NLTK resource downloaded successfully.")
except LookupError:
    print("NLTK resource already exists or download failed.")