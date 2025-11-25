
import os

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-16') as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading utf-16: {e}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading utf-8: {e}")

if __name__ == "__main__":
    print("--- backtest_results.txt ---")
    read_file_content('backtest_results.txt')
    print("\n--- backtest_results_v2.txt ---")
    read_file_content('backtest_results_v2.txt')
    print("\n--- quick_test_results_new.txt ---")
    read_file_content('quick_test_results_new.txt')
