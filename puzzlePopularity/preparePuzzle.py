import zstandard as zstd
import pandas as pd
import os


absolute_path = os.path.dirname(os.path.abspath(__file__))


# Path to the compressed file
compressed_file_path = absolute_path + r"\..\resources\lichess\lichess_db_puzzle.csv.zst"
# Path to the decompressed CSV file
decompressed_file_path = absolute_path + r"\..\resources\lichess\lichess_db_puzzle.csv"

# Save the filtered DataFrame to a new CSV file
filtered_file_path = absolute_path + r"\..\resources\lichess\filtered_lichess_db_puzzle.csv"

# Decompress the file
with open(compressed_file_path, 'rb') as compressed_file, open(decompressed_file_path, 'wb') as decompressed_file:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(compressed_file, decompressed_file)

# Load the decompressed CSV into a pandas DataFrame
df = pd.read_csv(decompressed_file_path)

# Replace empty strings with NaN (optional, if you have empty strings)
df.replace("", pd.NA, inplace=True)

# Filter by popularity (for example, keeping puzzles with popularity greater than a certain threshold)
popularity_threshold = 99  # You can set this to whatever value you like
rating_threshold = 2000
NbPlays_threshold = 40
# filtered_df = df[(df['Popularity'] > popularity_threshold) & (df['Rating'] < rating_threshold) & (df['NbPlays'] < NbPlays_threshold) & (df['OpeningTags'].notna()) & (df['OpeningTags'] != "")] 
filtered_df = df[(df['Popularity'] > popularity_threshold) & (df['Rating'] < rating_threshold) & (df['NbPlays'] < NbPlays_threshold) 
                 & (df['OpeningTags'].notna()) & (df['OpeningTags'] != "") & (~df['Themes'].str.contains('mateIn1', case=False, na=False))] 

filtered_df.to_csv(filtered_file_path, index=False)

print(f"Filtered data saved with lines {len(filtered_df)}")
