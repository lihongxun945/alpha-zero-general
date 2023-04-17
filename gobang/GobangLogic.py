'''
Author: MBoss
Date: Jan 17, 2018.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
class Board():
    def __init__(self, n):
        "Set up initial board configuration."
        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

   #def get_legal_moves(self, color):
   #    """Returns all the legal moves for the given color.
   #    (1 for white, -1 for black
   #    """
   #    moves = set()  # stores the legal moves.

   #    # Get all empty locations.
   #    for y in range(self.n):
   #        for x in range(self.n):
   #            if self[x][y] == 0:
   #                moves.add((x, y))
   #    return list(moves)

    #新方法，只返回2步以内有棋子的位置，这样能大幅减少搜索量"
    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    find = False
                    for i in [-2, -1, 0, 1, 2]:
                        if find:
                            break
                        for j in [-2, -1, 0, 1, 2]:
                            if (x+i>=0) & (x+i<self.n) & (y+j>=0) & (y+j<self.n):
                                if self[x+i][y+j] != 0:
                                    find = True
                                    break
                    if find:
                        moves.add((x, y))
        # 如果中间没人走，也要包括中间
        if self[int(self.n/2)][int(self.n/2)] == 0:
            moves.add((int(self.n/2), int(self.n/2)))
        return list(moves)

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        (x,y) = move
        assert self[x][y] == 0
        self[x][y] = color

