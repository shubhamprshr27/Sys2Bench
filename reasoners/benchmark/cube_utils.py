from __future__ import print_function
import numpy as np
import random
import re


'''
sticker indices:
       ┌──┬──┐
       │ 0│ 1│
       ├──┼──┤
       │ 2│ 3│
 ┌──┬──┼──┼──┼──┬──┬──┬──┐
 │16│17│ 8│ 9│ 4│ 5│20│21│
 ├──┼──┼──┼──┼──┼──┼──┼──┤
 │18│19│10│11│ 6│ 7│22│23│
 └──┴──┼──┼──┼──┴──┴──┴──┘
       │12│13│
       ├──┼──┤
       │14│15│
       └──┴──┘

face colors:
    ┌──┐
    │ 0│
 ┌──┼──┼──┬──┐
 │ 4│ 2│ 1│ 5│
 └──┼──┼──┴──┘
    │ 3│
    └──┘

moves:
[ U , U', U2, R , R', R2, F , F', F2, D , D', D2, L , L', L2, B , B', B2, x , x', x2, y , y', y2, z , z', z2]

'''

# move indices
moveInds = { \
  "U": 0, "U'": 1, "U2": 2, "R": 3, "R'": 4, "R2": 5, "F": 6, "F'": 7, "F2": 8, \
  "D": 9, "D'": 10, "D2": 11, "L": 12, "L'": 13, "L2": 14, "B": 15, "B'": 16, "B2": 17, \
  "x": 18, "x'": 19, "x2": 20, "y": 21, "y'": 22, "y2": 23, "z": 24, "z'": 25, "z2": 26 \
}

legal_moves = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]

# move definitions
moveDefs = np.array([ \
  [  2,  0,  3,  1, 20, 21,  6,  7,  4,  5, 10, 11, 12, 13, 14, 15,  8,  9, 18, 19, 16, 17, 22, 23], \
  [  1,  3,  0,  2,  8,  9,  6,  7, 16, 17, 10, 11, 12, 13, 14, 15, 20, 21, 18, 19,  4,  5, 22, 23], \
  [  3,  2,  1,  0, 16, 17,  6,  7, 20, 21, 10, 11, 12, 13, 14, 15,  4,  5, 18, 19,  8,  9, 22, 23], \
  [  0,  9,  2, 11,  6,  4,  7,  5,  8, 13, 10, 15, 12, 22, 14, 20, 16, 17, 18, 19,  3, 21,  1, 23], \
  [  0, 22,  2, 20,  5,  7,  4,  6,  8,  1, 10,  3, 12,  9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23], \
  [  0, 13,  2, 15,  7,  6,  5,  4,  8, 22, 10, 20, 12,  1, 14,  3, 16, 17, 18, 19, 11, 21,  9, 23], \
  [  0,  1, 19, 17,  2,  5,  3,  7, 10,  8, 11,  9,  6,  4, 14, 15, 16, 12, 18, 13, 20, 21, 22, 23], \
  [  0,  1,  4,  6, 13,  5, 12,  7,  9, 11,  8, 10, 17, 19, 14, 15, 16,  3, 18,  2, 20, 21, 22, 23], \
  [  0,  1, 13, 12, 19,  5, 17,  7, 11, 10,  9,  8,  3,  2, 14, 15, 16,  6, 18,  4, 20, 21, 22, 23], \
  [  0,  1,  2,  3,  4,  5, 10, 11,  8,  9, 18, 19, 14, 12, 15, 13, 16, 17, 22, 23, 20, 21,  6,  7], \
  [  0,  1,  2,  3,  4,  5, 22, 23,  8,  9,  6,  7, 13, 15, 12, 14, 16, 17, 10, 11, 20, 21, 18, 19], \
  [  0,  1,  2,  3,  4,  5, 18, 19,  8,  9, 22, 23, 15, 14, 13, 12, 16, 17,  6,  7, 20, 21, 10, 11], \
  [ 23,  1, 21,  3,  4,  5,  6,  7,  0,  9,  2, 11,  8, 13, 10, 15, 18, 16, 19, 17, 20, 14, 22, 12], \
  [  8,  1, 10,  3,  4,  5,  6,  7, 12,  9, 14, 11, 23, 13, 21, 15, 17, 19, 16, 18, 20,  2, 22,  0], \
  [ 12,  1, 14,  3,  4,  5,  6,  7, 23,  9, 21, 11,  0, 13,  2, 15, 19, 18, 17, 16, 20, 10, 22,  8], \
  [  5,  7,  2,  3,  4, 15,  6, 14,  8,  9, 10, 11, 12, 13, 16, 18,  1, 17,  0, 19, 22, 20, 23, 21], \
  [ 18, 16,  2,  3,  4,  0,  6,  1,  8,  9, 10, 11, 12, 13,  7,  5, 14, 17, 15, 19, 21, 23, 20, 22], \
  [ 15, 14,  2,  3,  4, 18,  6, 16,  8,  9, 10, 11, 12, 13,  1,  0,  7, 17,  5, 19, 23, 22, 21, 20], \
  [  8,  9, 10, 11,  6,  4,  7,  5, 12, 13, 14, 15, 23, 22, 21, 20, 17, 19, 16, 18,  3,  2,  1,  0], \
  [ 23, 22, 21, 20,  5,  7,  4,  6,  0,  1,  2,  3,  8,  9, 10, 11, 18, 16, 19, 17, 15, 14, 13, 12], \
  [ 12, 13, 14, 15,  7,  6,  5,  4, 23, 22, 21, 20,  0,  1,  2,  3, 19, 18, 17, 16, 11, 10,  9,  8], \
  [  2,  0,  3,  1, 20, 21, 22, 23,  4,  5,  6,  7, 13, 15, 12, 14,  8,  9, 10, 11, 16, 17, 18, 19], \
  [  1,  3,  0,  2,  8,  9, 10, 11, 16, 17, 18, 19, 14, 12, 15, 13, 20, 21, 22, 23,  4,  5,  6,  7], \
  [  3,  2,  1,  0, 16, 17, 18, 19, 20, 21, 22, 23, 15, 14, 13, 12,  4,  5,  6,  7,  8,  9, 10, 11], \
  [ 18, 16, 19, 17,  2,  0,  3,  1, 10,  8, 11,  9,  6,  4,  7,  5, 14, 12, 15, 13, 21, 23, 20, 22], \
  [  5,  7,  4,  6, 13, 15, 12, 14,  9, 11,  8, 10, 17, 19, 16, 18,  1,  3,  0,  2, 22, 20, 23, 21], \
  [ 15, 14, 13, 12, 19, 18, 17, 16, 11, 10,  9,  8,  3,  2,  1,  0,  7,  6,  5,  4, 23, 22, 21, 20]  \
])

# piece definitions
pieceDefs = np.array([ \
  [  0, 21, 16], \
  [  2, 17,  8], \
  [  3,  9,  4], \
  [  1,  5, 20], \
  [ 12, 10, 19], \
  [ 13,  6, 11], \
  [ 15, 22,  7], \
])

# OP representation from (hashed) piece stickers
pieceInds = np.zeros([58, 2], dtype=int)
pieceInds[50] = [0, 0]; pieceInds[54] = [0, 1]; pieceInds[13] = [0, 2]
pieceInds[28] = [1, 0]; pieceInds[42] = [1, 1]; pieceInds[ 8] = [1, 2]
pieceInds[14] = [2, 0]; pieceInds[21] = [2, 1]; pieceInds[ 4] = [2, 2]
pieceInds[52] = [3, 0]; pieceInds[15] = [3, 1]; pieceInds[11] = [3, 2]
pieceInds[47] = [4, 0]; pieceInds[30] = [4, 1]; pieceInds[40] = [4, 2]
pieceInds[25] = [5, 0]; pieceInds[18] = [5, 1]; pieceInds[35] = [5, 2]
pieceInds[23] = [6, 0]; pieceInds[57] = [6, 1]; pieceInds[37] = [6, 2]

# piece stickers from OP representation
pieceCols = np.zeros([7, 3, 3], dtype=int)
pieceCols[0, 0, :] = [0, 5, 4]; pieceCols[0, 1, :] = [4, 0, 5]; pieceCols[0, 2, :] = [5, 4, 0]
pieceCols[1, 0, :] = [0, 4, 2]; pieceCols[1, 1, :] = [2, 0, 4]; pieceCols[1, 2, :] = [4, 2, 0]
pieceCols[2, 0, :] = [0, 2, 1]; pieceCols[2, 1, :] = [1, 0, 2]; pieceCols[2, 2, :] = [2, 1, 0]
pieceCols[3, 0, :] = [0, 1, 5]; pieceCols[3, 1, :] = [5, 0, 1]; pieceCols[3, 2, :] = [1, 5, 0]
pieceCols[4, 0, :] = [3, 2, 4]; pieceCols[4, 1, :] = [4, 3, 2]; pieceCols[4, 2, :] = [2, 4, 3]
pieceCols[5, 0, :] = [3, 1, 2]; pieceCols[5, 1, :] = [2, 3, 1]; pieceCols[5, 2, :] = [1, 2, 3]
pieceCols[6, 0, :] = [3, 5, 1]; pieceCols[6, 1, :] = [1, 3, 5]; pieceCols[6, 2, :] = [5, 1, 3]

# useful arrays for hashing
hashOP = np.array([1, 2, 10])
pow3 = np.array([1, 3, 9, 27, 81, 243])
pow7 = np.array([1, 7, 49, 343, 2401, 16807])
fact6 = np.array([720, 120, 24, 6, 2, 1])

# get FC-normalized solved state
def initState():
  return np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])

'''

       ┌──┬──┐
       │ 0│ 1│
       ├──┼──┤
       │ 2│ 3│
 ┌──┬──┼──┼──┼──┬──┬──┬──┐
 │16│17│ 8│ 9│ 4│ 5│20│21│
 ├──┼──┼──┼──┼──┼──┼──┼──┤
 │18│19│10│11│ 6│ 7│22│23│
 └──┴──┼──┼──┼──┴──┴──┴──┘
       │12│13│
       ├──┼──┤
       │14│15│
       └──┴──┘
['Upper', 'Right', 'Front', 'Down', 'Left', 'Back']
'''
def getCube(s):
    cube_string = ""
    cube_string += "Upper:\n"  
    cube_string += "{} {}\n".format(s[0], s[1])  
    cube_string += "{} {}\n".format(s[2], s[3])  

    cube_string += "Front:\n"  
    cube_string += "{} {}\n".format(s[8], s[9])  
    cube_string += "{} {}\n".format(s[10], s[11])   

    cube_string += "Down:\n"  
    cube_string += "{} {}\n".format(s[12], s[13])  
    cube_string += "{} {}\n".format(s[14], s[15]) 

    cube_string += "Left:\n"  
    cube_string += "{} {}\n".format(s[16], s[17])  
    cube_string += "{} {}\n".format(s[18], s[19])   

    cube_string += "Right:\n"  
    cube_string += "{} {}\n".format(s[4], s[5])  
    cube_string += "{} {}\n".format(s[6], s[7]) 

    cube_string += "Back:\n"  
    cube_string += "{} {}\n".format(s[20], s[21])  
    cube_string += "{} {}\n".format(s[22], s[23])  

    return cube_string

# apply a move to a state
def doMove(s, move):
  # print(s)
  # print('move',move)
  # print('move2',moveDefs[move])
  return s[moveDefs[move]]

# apply a string sequence of moves to a state
def doAlgStr(s, alg):
  # print('',alg)
  moves = alg.split(" ")
  # print('moves',moves)
  for m in moves:
    if m in moveInds:
      s = doMove(s, moveInds[m])
  return s

# check if state is solved
def isSolved(s):
    for i in range(6):
        # 检查每个面的四个元素是否相同
        if not all(x == s[4 * i] for x in s[4 * i:4 * i + 4]):
            return False
    return True

# normalize stickers relative to a fixed DLB corner
def normFC(s):
  normCols = np.zeros(6, dtype=int)
  normCols[s[18] - 3] = 1
  normCols[s[23] - 3] = 2
  normCols[s[14]] = 3
  normCols[s[18]] = 4
  normCols[s[23]] = 5
  return normCols[s]

# get OP representation given FC-normalized sticker representation
def getOP(s):
  return pieceInds[np.dot(s[pieceDefs], hashOP)]

# get sticker representation from OP representation
def getStickers(sOP):
  s = np.zeros(24, dtype=np.int)
  s[[14, 18, 23]] = [3, 4, 5]
  for i in range(7):
    s[pieceDefs[i]] = pieceCols[sOP[i, 0], sOP[i, 1], :]
  return s

# get a unique index for the piece orientation state (0-728)
def indexO(sOP):
  return np.dot(sOP[:-1, 1], pow3)

# get a unique index for the piece permutation state (0-117648)
def indexP(sOP):
  return np.dot(sOP[:-1, 0], pow7)

# get a (gap-free) unique index for the piece permutation state (0-5039)
def indexP2(sOP):
  return np.dot([sOP[i, 0] - np.count_nonzero(sOP[:i, 0] < sOP[i, 0]) for i in range(6)], fact6)
  '''
  ps = np.arange(7)
  P = 0
  for i, p in enumerate(sOP[:, 0]):
    P += fact6[i] * np.where(ps == p)[0][0]
    ps = ps[ps != p]
  return P
  '''


def getface(state):
  faces = {}
  faces["Top"] = state[:4]
  faces["Right"] = state[4:8]
  faces["Front"] = state[8:12]
  faces["Down"] = state[12:16]
  faces["Left"] = state[16:20]
  faces["Back"] = state[20:24]
  return faces


def printfaces(facesdict):
  
  strings = """"""
  for face in ["Top", "Front", "Down", "Left", "Right", "Back"]:
    strings += face + ":\n"
    strings += "%s %s\n" % tuple(facesdict[face][:2])
    strings += "%s %s\n" % tuple(facesdict[face][2:])
  return strings
  

# get a unique index for the piece orientation and permutation state (0-3674159)
def indexOP(sOP):
  return indexO(sOP) * 5040 + indexP2(sOP)

# print state of the cube
def printCube(s):
  print("      ┌──┬──┐")
  print("      │ {}│ {}│".format(s[0], s[1]))
  print("      ├──┼──┤")
  print("      │ {}│ {}│".format(s[2], s[3]))
  print("┌──┬──┼──┼──┼──┬──┬──┬──┐")
  print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[16], s[17], s[8], s[9], s[4], s[5], s[20], s[21]))
  print("├──┼──┼──┼──┼──┼──┼──┼──┤")
  print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[18], s[19], s[10], s[11], s[6], s[7], s[22], s[23]))
  print("└──┴──┼──┼──┼──┴──┴──┴──┘")
  print("      │ {}│ {}│".format(s[12], s[13]))
  print("      ├──┼──┤")
  print("      │ {}│ {}│".format(s[14], s[15]))
  print("      └──┴──┘")


#self-defined useful functions:

def conduct_spin(cube_status):
  """
  input: cube current status: e.g. 1,1,1,1; 2,2,2,2; 3,3,3,3'; 4,4,4,4; 5,5,5,5,;6,6,6,6;
  output: flag(is valid?) , ans_str, num_list
  """
  
  legal_spins = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
  print("spin generate")
  try:
    orig_color_list = cube_status[:]
    operation = random.choice(legal_spins)
    current_color_list = doAlgStr(orig_color_list, operation)
    readable_res = "Original: \n" + getCube(orig_color_list) + "--(Operation:"+operation + ")-->\n" + "Current: \n" + getCube(current_color_list)
    return True, readable_res, current_color_list
  except:
    return False, None, None

def generate_spin(proposals, o_cube_status):
    """
    input : 
        proposals: input prompt string returned by LLM models (e.g. "[R U' F2]").
        o_cube_status: the original cube color status in string (e.g. '1,1,1,1; 2,2,2,2; ...').
    
    output: 
        proposals: the same input proposals string.
        o_cube_status: the cube status after applying the proposed spin operations.
    """
    # Extract moves from proposals (assuming proposals is like "[R U' F2]")
    moves = re.findall(r"[URF]2?'?", proposals)  # Find valid moves (U, U', U2, R, R', etc.)

    print(f"Extracted moves from proposal: {moves}")

    # Initialize current cube status
    current_cube_status = o_cube_status[:]
    
    # Apply each move to the cube status
    try:
        for move in moves:
            # Assuming doAlgStr applies the move to the cube status
            current_cube_status = doAlgStr(current_cube_status, move)
        
        # Generate readable output for the new cube status after the moves
        readable_result = f"Original Status: \n{getCube(o_cube_status)}\n" \
                          f"Applied Moves: {', '.join(moves)}\n" \
                          f"Current Status: \n{getCube(current_cube_status)}"
        
        print(f"Generated spin successfully: \n{readable_result}")
        
        return True, proposals, None
    except Exception as e:
        print(f"Error occurred while generating spin: {e}")
        return False, proposals, o_cube_status  # Return original status if something goes wrong


if __name__ == "__main__":
# get solved state
  s = initState()
  print(s)
  s = np.array([0, 0, 0, 0, 5, 5, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 5, 5])
  print(s)
#   printCube(s)
  # do some moves
  s = doAlgStr(s, "R")
  print("after", s)
  # printCube(s)
  # normalize stickers relative to DLB
  s = normFC(s)

  _, n, l= conduct_spin(s)

  print(n)