
def boards_to_tensor_batch(boards):
    """Convert a list of boards to a batch tensor"""
    batch_size = len(boards)
    tensor_batch = torch.zeros(batch_size, 12, 8, 8)
    
    for b_idx, board in enumerate(boards):
        tensor_batch[b_idx] = board_to_tensor(board)
    
    return tensor_batch


def get_valid_moves(board):
    """Get valid moves with caching"""
    return get_valid_moves_cached(board.fen())
