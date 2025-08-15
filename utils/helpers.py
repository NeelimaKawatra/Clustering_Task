import os

def get_file_from_upload(uploaded_file):
    """Convert uploaded file to temporary file path for backend processing"""
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    return None