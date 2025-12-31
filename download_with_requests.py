import requests
import os

# Download the infodynamics.jar file using requests library
url = "https://raw.githubusercontent.com/jlizier/jidt/master/infodynamics.jar"
filename = "infodynamics.jar"

try:
    print(f"Downloading {filename} from {url}...")
    
    # Set a reasonable timeout and stream the download
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()  # Check if the request was successful
        
        print(f"Response status: {response.status_code}")
        print(f"Content type: {response.headers.get('content-type')}")
        print(f"Content length: {response.headers.get('content-length')}")
        
        # Write the file in chunks
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Print progress
                    if total_size > 0:
                        percent = (downloaded_size / total_size) * 100
                        print(f"Downloaded: {downloaded_size}/{total_size} bytes ({percent:.1f}%)")
        
        # Verify the download
        if os.path.exists(filename):
            final_size = os.path.getsize(filename)
            print(f"\nDownload completed successfully!")
            print(f"File: {filename}")
            print(f"Size: {final_size} bytes")
            
            if total_size > 0 and final_size == total_size:
                print("✓ File size matches expected content length")
            else:
                print("⚠ File size doesn't match expected content length")
        else:
            print("\n❌ Error: File was not created!")
            
except requests.exceptions.HTTPError as e:
    print(f"\n❌ HTTP Error: {e}")
except requests.exceptions.ConnectionError as e:
    print(f"\n❌ Connection Error: {e}")
except requests.exceptions.Timeout as e:
    print(f"\n❌ Timeout Error: {e}")
except Exception as e:
    print(f"\n❌ Unexpected Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
