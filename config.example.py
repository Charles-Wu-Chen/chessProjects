# Example configuration file - copy to config.py and modify paths - this is for uploading to github without exposing local paths
LEELA_PATH = r'path/to/lc0.exe'
LEELA_WEIGHTS = r'path/to/weights.pb.gz'
STOCKFISH_PATH = r'path/to/stockfish.exe'

LEELA_OPTIONS = {
    'WeightsFile': LEELA_WEIGHTS,
    'UCI_ShowWDL': 'true'
}

STOCKFISH_OPTIONS = {
    'Threads': '10',
    'Hash': '4096'
} 