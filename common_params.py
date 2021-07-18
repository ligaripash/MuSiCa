
BB_SIZE = 142
INITIAL_POINTS_BB_SIZE = 142
ITERATIONS_COUNT = 3
SINGLE_POSE_ONLY = True
PYRAMID_HEIGHT = 3
INIT_POSE_COUNT = 1


#FLM group is defined by indexes into the 165 point FLM group
#PATCH group is a subset of FLM group (it contains indexes into this group)
# PATCH_INDEXES = [0, 117, 3, 114, 6, 111, 9, 108, 12, 105, 15, 102, 18, 55, 58, 61, 64, 100, 67, 70, 73, 76, 99, 42, 45,
#                  48, 51, 54, 41, 19, 21, 24, 26, 38, 30, 33, 35, 79, 81, 82, 83, 85, 155, 88, 152]

# 23 points
#PATCH_INDEXES = [0,  3, 6, 9, 12, 15, 18, 58, 64, 67, 73, 99, 45, 51, 41, 21, 26, 30, 35, 81, 83, 155, 152]
#FLM_INDEXES = range(165)


#68 points
#PATCH_INDEXES = [0, 4, 8, 12, 16, 17, 21, 22, 26, 36, 39, 42, 45, 31, 34, 48, 51, 54, 57]
PATCH_INDEXES = range(0, 98, 2)
#PATCH_INDEXES = range(98)
FLM_INDEXES = range(98)

# 10 Points
# FLM_INDEXES = [55, 58, 61, 64, 100, 67, 70, 73, 76, 99]
# PATCH_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FLM_COUNT = len(FLM_INDEXES)
PATCH_COUNT = len(PATCH_INDEXES)
def get_bb_size():
    return BB_SIZE

def get_initial_points_bb_size():
    return INITIAL_POINTS_BB_SIZE


MARGIN = 0.3