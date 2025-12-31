import urllib.request
import os

# Simple download with timeout
url = "https://raw.githubusercontent.com/jlizier/jidt/master/infodynamics.jar"
filename = "infodynamics.jar"

try:
    print("Starting simple download...")
    # Set timeout to 10 seconds
    response = urllib.request.urlopen(url, timeout=10)
    data = response.read()
    
    with open(filename, 'wb') as f:
        f.write(data)
    
    print(f"Downloaded {len(data)} bytes")
    
    if os.path.exists(filename):
        print(f"File saved as {filename}")
        print(f"File size: {os.path.getsize(filename)} bytes")
    else:
        print("Error: File not saved")
        
except Exception as e:
    print(f"Error: {e}")
