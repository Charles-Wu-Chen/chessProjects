import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions
import chardet
import chess


with open(functions.relativePathToAbsPath(r'\test\pgn\20250112\lichess_study_2024-2nd-half_round-7-akokbenjamin-wu-chen_by_wuchen1_2024.12.30.pgn'), 'rb') as f:
    result = chardet.detect(f.read())
    print(result)


# with open(functions.relativePathToAbsPath(r'\out\pgn\Kevin0817\Kevin _vs_Ariyathilaka,Saheli _2024.08.09.pgn'), 'r',  encoding='utf-8')as pgn_file:
#     # Parse the PGN file to get the first game
#     game = chess.pgn.read_game(pgn_file)