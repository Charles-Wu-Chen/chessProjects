import os
import sys
import argparse
from fenToImage import generate_images_from_csv, functions

def main(input_csv_path):
    # Convert relative path to absolute path
    csv_file_path = functions.relativePathToAbsPath(input_csv_path)
    
    # Check if the input CSV file exists, if not use the default
    if not os.path.exists(csv_file_path):
        print(f"Warning: Input file '{csv_file_path}' not found. Using default file.")
        csv_file_path = functions.relativePathToAbsPath(r'\out\pgn\test\comment\20240926231058move_details.csv')
    
    # Create output directory
    output_image_path = functions.relativePathToAbsPath(r'\out\pgn\test_fenToImage')
    
    # Ensure the output directory exists
    os.makedirs(output_image_path, exist_ok=True)
    
    # Generate images from the CSV file
    generate_images_from_csv(csv_file_path, output_image_path)
    
    print(f"Images generated and saved in: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chess board images from a CSV file.")
    parser.add_argument("--input_csv", help="Path to the input CSV file (e.g., 'out\\pgn\\0923Alice\\comment\\20240923040516move_details.csv')")
    
    args = parser.parse_args()
    
    main(args.input_csv if args.input_csv else r'\out\pgn\0923Alice\comment\20240923040516move_details.csv')