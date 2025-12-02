import requests
import json

def download_and_save_json(url: str, output_filename: str):
    try:
        resp = requests.get(url)
        resp.raise_for_status()  # HTTP error হলে exception তুলবে
    except requests.exceptions.RequestException as e:
        print("Error fetching URL:", e)
        return

    try:
        data = resp.json()  # JSON হিসেবে parse
    except ValueError as e:
        print("Error decoding JSON:", e)
        return

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_filename}")
    except IOError as e:
        print("Error writing file:", e)


if __name__ == "__main__":
    url = "https://www.mygov.bd/cache/service/search_service.json"
    output_file = "search_service_dump.json"
    download_and_save_json(url, output_file)

    
    file_path = "search_service_dump.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count_items = len(data)

    print("Total count:", count_items)
