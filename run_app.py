import os
import sys
import subprocess

def find_streamlit():
    """Find the streamlit executable in the Python installation"""
    base_dir = os.path.dirname(sys.executable)
    script_dir = os.path.join(base_dir, "Scripts")
    
    # Check common locations
    possible_paths = [
        os.path.join(script_dir, "streamlit.exe"),
        os.path.join(base_dir, "Scripts", "streamlit.exe"),
        os.path.join(base_dir, "bin", "streamlit"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Packages", 
                    "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0", 
                    "LocalCache", "local-packages", "Python313", "Scripts", "streamlit.exe")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found Streamlit at: {path}")
            return path
    
    return None

def run_streamlit_app():
    """Run the Streamlit app using the correct path"""
    streamlit_path = find_streamlit()
    
    if not streamlit_path:
        print("Error: Could not find the Streamlit executable.")
        print("Running with Python directly instead (may show warnings):")
        subprocess.run([sys.executable, "app.py"])
        return
    
    print("Starting Streamlit app...")
    # Run streamlit with the app
    subprocess.run([streamlit_path, "run", "app.py"])

if __name__ == "__main__":
    run_streamlit_app()
