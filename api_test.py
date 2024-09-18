import requests
import sys
import os

def test_api(image_path):
    url = "http://localhost:8000/api/classify"

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"Class Label: {result['class_label']}")
            print(f"Confidence: {result['confidence']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python api_test.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    test_api(image_path)
