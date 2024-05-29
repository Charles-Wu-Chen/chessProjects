import Bitboard
import chess

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions


def movePieceToSquare(fen: str, oldSquare: str, newSquare: str, changeMove: bool = True) -> str:
    """
    This function moves a piece in a FEN to a new square and returns the new FEN.
    fen: str
        The FEN string of the starting position
    oldSquare, newSquare: str:
        The original square of the piece and the destination square
    changeMove: bool
        If this is True, the new FEN will have the other side to move
    return -> str
        The new FEN string
    """
    board = Bitboard.Bitboard()
    board.setBoardFEN(fen)
    if (newBoard := board.moveToNewSquare(oldSquare, newSquare)):
        newPos = newBoard.toFEN()
        index = fen.index(' ')
        newFEN = f'{newPos}{fen[index:]}'
        if changeMove:
            index = newFEN.index(' ')+1
            if newFEN[index] == 'b':
                newFEN = f'{newFEN[:index]}w{newFEN[index+1:]}'
            else:
                newFEN = f'{newFEN[:index]}b{newFEN[index+1:]}'
        return newFEN
    return None


def findIdealSquare(fen: str, startSquare: str, sf: chess.engine, changeMove: bool = True) -> str:
    """
    This function finds the best square for the piece standing on the startSquare
    fen: str
        The position
    startSquare: str
        The starting square of the piece
    sf: chess.engine
        Stockfish as a configured chess engine in order to evaluate the posioons
    changeMove: bool
        If this is True, the new position will have the other side to move
    return -> str
        The ideal square for the piece
    """
    time = 3
    board = chess.Board(fen)
    startCP = int(str(sf.analyse(board, chess.engine.Limit(time=time))['score'].white()))
    improvement = 0
    white = board.turn
    bestSquare = startSquare

    bitboard = Bitboard.Bitboard()
    bitboard.setBoardFEN(fen)
    initialMaterialDiff = bitboard.materialDiff()
    evalChanges = dict()

    if bitboard.squareIsEmpty(startSquare):
        return None
    for f in 'abcdefgh':
        for r in range(1, 9):
            newSquare = f'{f}{r}'
            if bitboard.squareIsEmpty(newSquare) and newSquare != startSquare:
                newPos = movePieceToSquare(fen, startSquare, newSquare, changeMove)
                board = chess.Board(newPos)
                info = sf.analyse(board, chess.engine.Limit(time=time))
                matDiffAfter = materialDiffAfterPV(board, info['pv'])
                newCP = int(str(info['score'].white()))
                if white:
                    factor = 1
                else:
                    factor = -1
                gap = factor * (newCP - startCP)
                if factor*(matDiffAfter - initialMaterialDiff) <= 2:
                    print(newSquare, gap, matDiffAfter)
                    evalChanges[newSquare] = gap
    for sq, ev in evalChanges.items():
        if ev > improvement:
            bestSquare = sq
            improvement = ev
    return bestSquare


def materialDiffAfterPV(board: chess.Board, pv: list) -> int:
    """
    This function calculates the material difference at the end of the PV
    board: chess.Board
        The board in the starting position
    pv: list
        A list of the PV moves, attained from an engine analysis
    return -> int
        The material difference at the end of the PV
    """
    board2 = board
    for move in pv:
        board2.push(move)
    bb = Bitboard.Bitboard()
    bb.setBoardFEN(board2.fen())
    return bb.materialDiff()


if __name__ == '__main__':
    sf = functions.configureEngine('stockfish', {'Threads': '10', 'Hash': '8192'})
    fen = '3rnrk1/2qn1pbp/1p4p1/2p1p3/4P3/4B1PP/1PPNQPB1/R4RK1 w - - 0 18'
    # print(fen)
    # print(movePieceToSquare(fen, 'd2', 'b5'))
    # print(findIdealSquare(fen, 'd2', sf, True))
    fen2 = '2bqr1k1/r4pnp/1p3bp1/pPp1p3/2P1P3/3P1NP1/PBQ3BP/4RRK1 w - - 1 20'
    fens = ['r2q1rk1/ppb2ppp/2p1bn2/5p2/2NP4/4P1P1/PPQ1NPBP/R4RK1 w - - 3 13',
            'r2qrnk1/pp3pb1/3p4/2pPp1p1/2P1P3/2N2PBR/PPQ2P2/2KR4 w - - 1 19',
            'r3r1k1/pb2qppp/1n6/2b3PN/4p1QP/2N1P3/1P1B1P2/R3K2R b K - 1 21']
    squares = ['c4', 'c3', 'b6']
    # print(findIdealSquare('r1bq1rk1/1p3ppp/p3pb2/4N3/PnBP4/8/1P2QPPP/R1BR2K1 w - - 0 15', 'a1', sf, True))
    print(findIdealSquare('r1b1k2r/pp1nqpp1/3p1n1p/2pPp3/P1P1P2P/2P2N2/2Q1BPP1/R1B1K2R b KQkq - 1 11', 'e8', sf, True))
    # for fen, square in zip(fens, squares):
    #     print(findIdealSquare(fen, square, sf, True))
    sf.quit()
