
import os

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-16') as f:
            print(f.read())
    except Exception:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    read_file_content('quick_test_results_new.txt')
