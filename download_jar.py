import urllib.request
import os
import sys

# Download the infodynamics.jar file
try:
    print("Starting download of infodynamics.jar...")
    print(f"Python version: {sys.version}")
    
    # Try with raw.githubusercontent.com
    url1 = "https://raw.githubusercontent.com/jlizier/jidt/master/infodynamics.jar"
    # Try with GitHub raw URL
    url2 = "https://github.com/jlizier/jidt/raw/master/infodynamics.jar"
    
    filename = "infodynamics.jar"
    
    # Try first URL
    try:
        print(f"\nTrying URL 1: {url1}")
        response = urllib.request.urlopen(url1, timeout=30)
        print(f"Response code: {response.getcode()}")
        print(f"Content type: {response.getheader('Content-Type')}")
        print(f"Content length: {response.getheader('Content-Length')}")
        
        # Read and write the file
        with open(filename, 'wb') as f:
            f.write(response.read())
        
        if os.path.exists(filename):
            filesize = os.path.getsize(filename)
            print(f"Download successful from URL 1!")
            print(f"File: {filename}")
            print(f"Size: {filesize} bytes")
        else:
            print("Error: File was not created from URL 1!")
            
    except Exception as e1:
        print(f"Error with URL 1: {e1}")
        
        # Try second URL if first fails
        print(f"\nTrying URL 2: {url2}")
        try:
            response = urllib.request.urlopen(url2, timeout=30)
            print(f"Response code: {response.getcode()}")
            print(f"Content type: {response.getheader('Content-Type')}")
            print(f"Content length: {response.getheader('Content-Length')}")
            
            # Read and write the file
            with open(filename, 'wb') as f:
                f.write(response.read())
            
            if os.path.exists(filename):
                filesize = os.path.getsize(filename)
                print(f"Download successful from URL 2!")
                print(f"File: {filename}")
                print(f"Size: {filesize} bytes")
            else:
                print("Error: File was not created from URL 2!")
                
        except Exception as e2:
            print(f"Error with URL 2: {e2}")
            raise

except Exception as e:
    print(f"\nFinal error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
