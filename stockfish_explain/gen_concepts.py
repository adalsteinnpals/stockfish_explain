import chess
import numpy as np
import copy
import pandas as pd


def connected_rooks(np_board, rook_symbol = 'R'):
    rooks = np.where(np_board == rook_symbol)
    if len(rooks[0]) > 1: # more than one rook 
        if (rooks[0][0] == rooks[0][1]) | (rooks[1][0] == rooks[1][1]): # same file
            rook_line = np_board[min(rooks[0]):(max(rooks[0])+1), min(rooks[1]):(max(rooks[1])+1)]
            return ((rook_line == '.') | (rook_line == rook_symbol)).all()
    return False


def can_check(board):
    for move in board.legal_moves:
        board_ = copy.copy(board)
        if board_.gives_check(move):
            return True
    return False



def can_fork(board):
    turn = board.turn
    
    for move in board.legal_moves:
        board_ = copy.copy(board)
        board_.push(chess.Move.from_uci(str(move)))
        if is_knight_forking(board_, turn, verbose = True):
            return True
    return False



def is_knight_forking(board, turn=None, verbose = False):
    
    if turn is None:
        turn = board.turn

    high_value_pieces_white = ['K','Q','R']
    high_value_pieces_black = ['k','q','r']

    if turn:
        high_value_pieces = high_value_pieces_black
    else:
        high_value_pieces = high_value_pieces_white



    piece_map = board.piece_map()
    knight_string = 'N' if turn else 'n'

    knight_positions = [d[0] for d in piece_map.items() if str(d[1]) == knight_string]
    

    if len(knight_positions) > 0:
        for kn_pos in knight_positions:
            attacking = []
            if not board.is_pinned(board.turn, kn_pos):
                for square in board.attacks(kn_pos):
                    piece = board.piece_at(square)
                    if str(piece) in high_value_pieces:
                        attacking.append(piece)
                if len(attacking) > 1:
                    return True
    return False








def create_custom_concepts(board):
    
    
    
    # Bishop Pair
    concepts = {}
    pieces = [val.symbol() for val in board.piece_map().values()]
    concepts['white_bishop_pair'] = int(pieces.count('B') == 2)
    concepts['black_bishop_pair'] = int(pieces.count('b') == 2)
    
    # Knight Pair
    concepts['white_knight_pair'] = int(pieces.count('N') == 2)
    concepts['black_knight_pair'] = int(pieces.count('n') == 2)
    

    # Double Pawn
    np_board = np.array([x.split(' ') for x in str(board).split('\n')])
    concepts['white_double_pawn'] = int(any((np_board == 'P').sum(axis = 0) > 1))
    concepts['black_double_pawn'] = int(any((np_board == 'p').sum(axis = 0) > 1))

    # Isolated Pawns
    pos_any_white = ((np_board == 'P').sum(axis=0) > 0)
    isolated_pawns_white = np.where(pos_any_white & ~np.roll(pos_any_white, 1) & ~np.roll(pos_any_white, -1))[0]
    pos_any_black = ((np_board == 'p').sum(axis=0) > 0)
    isolated_pawns_black = np.where(pos_any_black & ~np.roll(pos_any_black, 1) & ~np.roll(pos_any_black, -1))[0]
    concepts['white_isolated_pawns'] = int(len(isolated_pawns_white) > 0)
    concepts['black_isolated_pawns'] = int(len(isolated_pawns_black) > 0)

    # Connected rooks
    concepts['white_connected_rooks'] = int(connected_rooks(np_board, rook_symbol='R'))
    concepts['black_connected_rooks'] = int(connected_rooks(np_board, rook_symbol='r'))


    # Open Files
    no_pawns = (((np_board == 'p') | (np_board == 'P')).sum(axis=0) == 0)
    rook_or_queen_black = ((np_board == 'r') | (np_board == 'q')).any(axis=0)
    rook_or_queen_white = ((np_board == 'R') | (np_board == 'Q')).any(axis=0)

    concepts['white_has_control_of_open_file'] = any(no_pawns & ~rook_or_queen_black & rook_or_queen_white)
    concepts['black_has_control_of_open_file'] = any(no_pawns & rook_or_queen_black & ~rook_or_queen_white)
    concepts['has_contested_open_file'] = any(no_pawns & rook_or_queen_black & rook_or_queen_white)

    # Forking
    concepts['is_forking'] = is_knight_forking(board)
    concepts['can_fork'] = can_fork(board)
    
    # Checking
    concepts['can_check'] = can_check(board)


    return concepts
