import os
import sys
import shutil
from glob import glob
import traceback
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions
from functions import configureEngine, analysisCPnWDL
import fenToImage
import findMistake

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_directories(input_directory, comment_directory, split_directory):
    # Copy all .pgn files from comment_directory to input_directory
    pgn_files = glob(os.path.join(split_directory, "*.pgn"))
    for pgn_file in pgn_files:
        shutil.copy(pgn_file, input_directory)
        logger.info(f"Copied {pgn_file} to {input_directory}")

    # Clear the comment directory
    if os.path.exists(comment_directory):
        for filename in os.listdir(comment_directory):
            file_path = os.path.join(comment_directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    logger.info(f"Cleared comment directory: {comment_directory}")
    # Clear the split directory
    if os.path.exists(split_directory):
        for filename in os.listdir(split_directory):
            file_path = os.path.join(split_directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    logger.info(f"Cleared input directory: {split_directory}")
                
def main():
    logger.info("Starting main execution of test_FindMistake.py")
    
    input_folder = r"\out\pgn\test_findMistake"
    player_name = "alice"
    mistake_value = 100

    logger.info(f"Input folder: {input_folder}, Player name: {player_name}, Mistake value: {mistake_value}")

    input_directory = functions.relativePathToAbsPath(input_folder)
    output_directory = functions.relativePathToAbsPath(input_folder + "\\out")
    split_directory = functions.relativePathToAbsPath(input_folder + "\\split")
    comment_directory = functions.relativePathToAbsPath(input_folder + "\\comment")
    prepare_directories(input_directory, comment_directory, split_directory)

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(split_directory, exist_ok=True)
    os.makedirs(comment_directory, exist_ok=True)


    

    logger.info("Starting to split PGN files")
    raw_pgn_files = glob(os.path.join(input_directory, "*.pgn"))
    for pgn_file in raw_pgn_files:
        logger.info(f"Splitting file: {pgn_file}")
        findMistake.split_pgn_file(pgn_file, output_directory)
        shutil.move(pgn_file, split_directory)

    logger.info("Configuring chess engines")
    op = {'WeightsFile': r'K:\leela\lc0-v0.30.0-windows-gpu-nvidia-cudnn\791556.pb.gz', 'UCI_ShowWDL': 'true'}
    leela = configureEngine(r'K:\leela\lc0-v0.30.0-windows-gpu-nvidia-cudnn\lc0.exe', op)
    sf = configureEngine(r'K:\github\stockfish-windows-x86-64\stockfish\stockfish-windows-x86-64.exe', {'Threads': '10', 'Hash': '4096'})

    pgn_files = glob(os.path.join(output_directory, "*.pgn"))
    logger.info(f"Found {len(pgn_files)} PGN files after splitting")

    try:
        for pgn_file in pgn_files:
            logger.info(f"Processing file: {pgn_file}")
            try:
                findMistake.make_comments(pgn_file, os.path.join(comment_directory, os.path.basename(pgn_file)), analysisCPnWDL, 5000, leela, sf, True)
                os.makedirs(os.path.join(split_directory, "done"), exist_ok=True)
                shutil.move(pgn_file, os.path.join(split_directory, "done", os.path.basename(pgn_file)))
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
        
        logger.info("Starting to process PGN folder")
        detail_file_name = findMistake.process_pgn_folder(comment_directory, sf, player_name, mistake_value)
        logger.info(f"Finished processing PGN folder. Output file: {detail_file_name}")
    finally:
        logger.info("Quitting chess engines")
        sf.quit()
        leela.quit()

    logger.info("Generating chess board images")
    csv_file_path = os.path.join(comment_directory, detail_file_name)
    output_image_path = os.path.join(comment_directory, "img")
    os.makedirs(output_image_path, exist_ok=True)
    fenToImage.generate_images_from_csv(csv_file_path, output_image_path)

    logger.info("Finished main execution of test_FindMistake.py")

if __name__ == "__main__":
    main()