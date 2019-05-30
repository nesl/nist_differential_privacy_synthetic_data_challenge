"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import numpy as np


# Methods to identify the distinct set of values for each column

def valid_values_from_metdata(metadata, feat_name):
    """ collect distinct set of values according to the JSON metada file """
    return {
        k: v for k, v in enumerate(list(range(metadata[feat_name]['maxval']+1)))}


def valid_values_from_groundtruth(gt_df, feat_name):
    """ collect distinct set of values according from the input data.
    WE DO this ONLY for Geographic (state-dependent columns)
    """
    return {
        v: k for k, v in enumerate(gt_df[feat_name].unique().tolist())}


def valid_values_from_list(l):
    """
    Define distinct set of values according to a list we collected from the
    codebook file.
    """
    return {k: v for v, k in enumerate(l)}


def preprocess_nist_data(input_file, meta_file, subsample=False):
    """
    Applies pre-processing to the input data.

    returns the following
    original_df: the input data in the original format.
    input_df: the input data after beeing formatted.
    metadata: the metadata dictionary.
    subsample: this argument is True only when we debug the model to work on samller datasets.
    """
    original_df = pd.read_csv(input_file).astype(np.int32)
    if subsample:
        original_df = original_df.sample(100000)
    metadata = json.loads(open(meta_file).read())
    columns_list = original_df.columns.tolist()

    # Make it easy iso perform one hot encoding
    # Mappings areisbased on the codebook file.
    col_maps = {
        'SPLIT': {0: 0, 1: 1},  # from codebook
        'OWNERSHP': {0: 0, 1: 1, 2: 2},  # from codebook
        'VETWWI': {0: 0, 1: 1, 2: 2},  # from codebook
        'RESPONDT': {0: 0, 1: 1, 2: 2},  # from codebook
        'LABFORCE': {0: 0, 1: 1, 2: 2},  # from codebook
        'SLREC': {1: 0, 2: 1},  # from codebook
        'SSENROLL': {0: 0, 1: 1, 2: 2},  # from codebook
        'SPANNAME': {0: 0, 1: 1, 2: 2, 9: 3},  # from codebook
        'SCHOOL': {0: 0, 1: 1, 2: 2, 9: 3},  # from codebook
        'URBAN': {0: 0, 1: 1, 2: 2},  # from codebook
        'FARM': {0: 0, 1: 1, 2: 2},  # from codebook
        'SEX': {1: 0, 2: 1},  # from codebook
        'METRO': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},  # from codebook
        'EMPSTAT': {0: 0, 1: 1, 2: 2, 3: 3},  # from codebook
        'HISPAN': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5},  # from codebook
        'CITIZEN': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # from codebook
        'NATIVITY': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # from codebook
        'WARD': valid_values_from_metdata(metadata, 'WARD'),  # from specs file
        # from codebook
        'RACE': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8},
        # from codebook
        'WKSWORK2': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
        'MARST': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},  # from codebook
        'NCHLT5': {i: i for i in range(10)},  # from codebook
        'GQ': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},  # from specs file
        'VETPER': {i: i for i in range(9)},  # from codebook
        'HISPRULE': {i: i for i in range(9)},  # from codebook
        'VET1940': {0: 0, 1: 1, 2: 2, 8: 3},  # from codebook
        'UCLASSWK': {i: i for i in range(9)},  # from codebook
        'HRSWORK2': {i: i for i in range(9)},  # from codebook
        'SAMESEA5': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5},  # from codebook
        'SAMEPLAC': {0: 0, 1: 1, 2: 2, 9: 3},  # from codebook
        'GQTYPE': {i: i for i in range(10)},  # from codebook
        'VETCHILD': {0: 0, 1: 1, 2: 2, 8: 3, 9: 4},  # from codebook
        'CLASSWKR': {0: 0, 1: 1, 2: 2, 9: 3},  # from codebook
        'VETSTAT': {0: 0, 1: 1, 2: 2, 9: 3},  # from codebook
        'INCNONWG': {0: 0, 1: 1, 2: 2, 9: 3},  # from codebook
        'MARRNO': {i: i for i in range(10)},  # from codebook
        'MIGTYPE5': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5},  # from codebook
        # from codebook
        'MIGRATE5': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 8: 5, 9: 6},
        # from codebook
        'OWNERSHPD': {0: 0, 10: 1, 11: 2, 12: 3, 13: 4, 20: 5, 21: 6, 22: 7},
        # from specs file
        'FAMSIZE': valid_values_from_metdata(metadata, 'FAMSIZE'),
        'SIZEPL': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                   10: 10, 20: 11, 30: 12, 40: 13, 50: 14, 60: 15,  70: 16, 80: 17, 90: 18},  # from codebook
        'EMPSTATD': {0: 0, 10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 20: 7,
                     21: 8, 22: 9, 30: 10, 31: 11, 32: 12, 33: 13, 34: 14},  # from codebook
        # values are in range[1, 52]
        'WKSWORK1': valid_values_from_metdata(metadata, 'WKSWORK1'),

        # SEA is one of the geographic values, we are allowd to inspect their set of values from ground truth.
        # SEA is missing for codebook. Coloardo values are from https://usa.ipums.org/usa/volii/seacodes.shtml
        'SEA': valid_values_from_groundtruth(original_df, 'SEA'),

        # from specs file
        'OCCSCORE': valid_values_from_metdata(metadata, 'OCCSCORE'),

        # from metadata
        'AGEMARR': valid_values_from_metdata(metadata, 'AGEMARR'),
        'MIGRATE5D': {0: 0, 10: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 30: 8, 31: 9,
                      32: 10, 33: 11, 40: 12, 80: 13, 90: 14},  # from codebook
        # from spec file
        'MTONGUE':  valid_values_from_metdata(metadata, 'MTONGUE'),
        'SEI':  valid_values_from_metdata(metadata, 'SEI'),  # from spec file
        'CLASSWKRD':  {0: 0, 10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 20: 6, 21: 7,
                       22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 27: 13, 28: 14, 29: 15,
                       98: 16, 99: 17},  # from codebook
        # from spec file
        'HRSWORK1':  valid_values_from_metdata(metadata, 'HRSWORK1'),
        'HIGRADE':  {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
                     12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21,
                     22: 22, 23: 23, 99: 24},  # from codebook file
        'EDUC': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                 8: 8, 9: 9, 10: 10, 11: 11, 99: 12},  # from codebook
        # from codebook
        'VETSTATD': {0: 0, 10: 1, 11: 2, 12: 3, 13: 4, 20: 5, 21: 6, 22: 7, 23: 8, 99: 9},
        'CHBORN': valid_values_from_metdata(metadata, 'CHBORN'),  # from specs
        'GQFUNDS': {0: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 21: 7, 22: 8, 23: 9,
                    24: 10, 25: 11, 99: 12},  # from codebook
        'AGEMONTH': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                     11: 11, 12: 12, 98: 13, 99: 14},  # from codebook
        'AGE': valid_values_from_metdata(metadata, 'AGE'),  # max val is 108
        'HISPAND': valid_values_from_list([0, 100, 102, 103, 104, 105, 106, 107,
                                           200, 300, 401, 402, 411, 412, 413, 414, 415, 416, 417, 420, 421, 422, 423, 424,
                                           425, 426, 427, 428, 429, 430, 431, 450, 451, 452, 453, 454, 455, 456, 457, 458,
                                           459, 460, 465, 470, 480, 490, 491, 492, 493, 494, 495, 496, 498, 499, 900]),  # from codebook
        'RACED': valid_values_from_list(   # from codebook
            [100, 110, 120, 130, 140, 150, 200, 210, 300, 302,
             303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
             313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
             323, 324, 325, 326, 328, 329, 330, 350, 351, 352,
             353, 354, 355, 356, 357, 358, 359, 360, 361, 362,
             370, 371, 372, 373, 374, 375, 379, 398, 399, 400,
             410, 420, 500, 600, 610, 620, 630, 631, 632, 634,
             640, 641, 642, 643, 650, 651, 652, 653, 660, 661,
             662, 663, 664, 665, 666, 667, 668, 669, 670, 671,
             672, 673, 674, 675, 676, 677, 678, 679, 680, 681,
             682, 683, 684, 685, 686, 687, 688, 689, 690, 691,
             692, 698, 699, 700, 801, 802, 810, 811, 812, 813,
             814, 815, 816, 817, 818, 819, 820, 821, 822, 823,
             824, 825, 826, 827, 830, 831, 832, 833, 834, 835,
             836, 837, 838, 840, 841, 842, 845, 850, 851, 852,
             853, 854, 855, 856, 860, 861, 862, 863, 864, 865,
             866, 867, 868, 869, 880, 881, 882, 883, 884, 885,
             886, 887, 890, 891, 892, 893, 899, 901, 902, 903,
             904, 905, 906, 907, 910, 911, 912, 913, 914, 915,
             916, 917, 920, 921, 922, 923, 925, 930, 931, 932,
             933, 934, 935, 940, 941, 942, 943, 944, 949, 950,
             951, 952, 953, 954, 955, 960, 961, 962, 963, 964,
             970, 971, 972, 973, 974, 975, 976, 980, 981, 982,
             983, 984, 985, 986, 989, 990, 991, 996]),
        # TODO(malzantot): check
        # geographic
        'SUPDIST': valid_values_from_metdata(metadata, 'SUPDIST'),
        # coloardo values
        'METAREA': valid_values_from_groundtruth(original_df, 'METAREA'),
        # from specs file
        'PRESGL': valid_values_from_metdata(metadata, 'PRESGL'),

        # from codebook
        'MBPL': valid_values_from_list([0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 90, 99, 100, 105, 110, 115, 120, 150, 155, 160, 199, 200, 210, 250, 260, 299, 300, 400, 401, 402, 403, 404, 405, 410, 411, 412, 413, 414, 419, 420, 421, 422, 423, 424, 425, 426, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 465, 499, 500, 501, 502, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 599, 600, 700, 710, 900, 950, 997, 999]),

        # from codebook
        'FBPL': valid_values_from_list([0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 90, 99, 100, 105, 110, 115, 120, 150, 155, 160, 199, 200, 210, 250, 260, 299, 300, 400, 401, 402, 403, 404, 405, 410, 411, 412, 413, 414, 419, 420, 421, 422, 423, 424, 425, 426, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 465, 499, 500, 501, 502, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 599, 600, 700, 710, 900, 950, 997, 999]),

        # from codebook
        'BPL': valid_values_from_list([0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 90, 99, 100, 105, 110, 115, 120, 150, 155, 160, 199, 200, 210, 250, 260, 299, 300, 400, 401, 402, 403, 404, 405, 410, 411, 412, 413, 414, 419, 420, 421, 422, 423, 424, 425, 426, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 465, 499, 500, 501, 502, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 599, 600, 700, 710, 900, 950, 997, 999]),

        # from codebook
        'IND1950': valid_values_from_list([0, 105, 116, 126, 206, 216, 226, 236, 239, 246, 306, 307, 308, 309, 316, 317, 318, 319, 326, 336, 337, 338, 346, 347, 348, 356, 357, 358, 367, 376, 377, 378, 379, 386, 387, 388, 399, 406, 407, 408, 409, 416, 417, 418, 419, 426, 429, 436, 437, 438, 439, 446, 448, 449, 456, 457, 458, 459, 466, 467, 468, 469, 476, 477, 478, 487, 488, 489, 499, 506, 516, 526, 527, 536, 546, 556, 567, 568, 578, 579, 586, 587, 588, 596, 597, 598, 606, 607, 608, 609, 616, 617, 618, 619, 626, 627, 636, 637, 646, 647, 656, 657, 658, 659, 667, 668, 669, 679, 686, 687, 688, 689, 696, 697, 698, 699, 716, 726, 736, 746, 756, 806, 807, 808, 816, 817, 826, 836, 846, 847, 848, 849, 856, 857, 858, 859, 868, 869, 879, 888, 896, 897, 898, 899, 906, 916, 926, 936, 946, 976, 979, 980, 982, 983, 984, 986, 987, 991, 995, 997, 998, 999]),

        # State enconomic area, the list of all SEA codes are from https://usa.ipums.org/usa/volii/seacodes.shtml
        'MIGSEA5': valid_values_from_list(list(range(503)) + list(range(990, 998))),

        # Occ codes in 1940 cnesus from https://usa.ipums.org/usa/volii/occ1940.shtml
        'OCC': valid_values_from_list([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 98, 99, 100, 102,
            104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128,
            130, 132, 134, 136, 156, 200, 210, 220, 222, 224, 226, 231, 236, 240,
            242, 244, 246, 248, 250, 252, 254, 256, 258, 266, 270, 272, 274, 276, 278,
            280, 282, 284, 286, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316,
            318, 320, 322, 324, 326, 327, 328, 330, 332, 334, 336, 338, 340, 342,
            344, 346, 348, 350, 352, 354, 356, 358, 260, 360, 362, 364, 366, 368, 370, 372, 374,
            376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402,
            404, 406, 408, 410, 412, 414, 416, 418, 420, 430, 432, 434, 436, 438,
            440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470,
            472, 474, 476, 478, 480, 482, 484, 486, 488, 496, 500, 510, 520, 600, 602, 604, 606, 608,
            610, 612, 614, 700, 710, 712, 714, 720, 730, 732, 740, 750, 760, 770, 780,
            790, 792, 794, 796, 798, 844, 866, 888, 902, 904, 906, 908, 910, 988, 900, 995, 996,
            998, 999
        ]),
        # from codebook
        'UOCC95': valid_values_from_list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 123, 200, 201, 203, 204, 205, 210, 230, 240, 250, 260, 270, 280, 290, 300, 301, 302, 304, 305, 310, 320, 321, 322, 325, 335, 340, 341, 342, 350, 360, 365, 370, 380, 390, 400, 410, 420, 430, 450, 460, 470, 480, 490, 500, 501, 502, 503, 504, 505, 510, 511, 512, 513, 514, 515, 520, 521, 522, 523, 524, 525, 530, 531, 532, 533, 534, 535, 540, 541, 542, 543, 544, 545, 550, 551, 552, 553, 554, 555, 560, 561, 562, 563, 564, 565, 570, 571, 572, 573, 574, 575, 580, 581, 582, 583, 584, 585, 590, 591, 592, 593, 594, 595, 600, 601, 602, 603, 604, 605, 610, 611, 612, 613, 614, 615, 620, 621, 622, 623, 624, 625, 630, 631, 632, 633, 634, 635, 640, 641, 642, 643, 644, 645, 650, 660, 661, 662, 670, 671, 672, 673, 674, 675, 680, 681, 682, 683, 684, 685, 690, 700, 710, 720, 730, 731, 732, 740, 750, 751, 752, 753, 754, 760, 761, 762, 763, 764, 770, 771, 772, 773, 780, 781, 782, 783, 784, 785, 790, 810, 820, 830, 840, 910, 920, 930, 940, 950, 960, 970, 975, 980, 982, 983, 984, 986, 987, 995, 997, 999]),
        # from codebook
        'UIND': valid_values_from_list(list(range(132))+[995, 996, 997, 998, 999]),
        # from codebook, assuming same set as UIND
        'IND': valid_values_from_list(list(range(132))+[995, 996, 997, 998, 999]),
        # from spec file. probably use an int ??
        'DURUNEMP': valid_values_from_metdata(metadata, 'DURUNEMP'),
        # from codebook
        'GQTYPED': valid_values_from_list([0, 10, 20, 100, 200, 210, 211, 212, 213, 220, 221, 230, 240, 250, 260, 300, 400, 410, 411, 412, 413, 420, 421, 430, 431, 432, 440, 441, 450, 451, 452, 460, 461, 470, 471, 472, 480, 481, 482, 491, 492, 493, 494, 495, 496, 500, 501, 502, 600, 601, 602, 603, 604, 700, 701, 800, 801, 802, 803, 804, 900, 901, 910, 911, 912, 913, 914, 920, 921, 922, 923, 924, 931, 932, 933, 934, 935, 936, 937, 940, 941, 942, 943, 944, 945, 946, 947, 948, 950, 955, 960, 999]),
        # from codebook
        'MIGPLAC5': valid_values_from_list([0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66, 67, 68, 99, 100, 105, 110, 115, 119, 120, 150, 151, 152, 155, 160, 199, 200, 211, 212, 213, 214, 215, 216, 217, 218, 219, 250, 260, 261, 262, 263, 264, 266, 267, 305, 310, 315, 320, 325, 330, 345, 350, 360, 365, 370, 390, 400, 401, 402, 404, 405, 410, 411, 412, 413, 414, 415, 420, 421, 422, 423, 424, 425, 426, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 450, 451, 452, 453, 454, 455, 456, 457, 460, 461, 462, 465, 496, 498, 499, 500, 501, 502, 510, 511, 512, 513, 514, 515, 516, 517, 518, 520, 521, 525, 522, 523, 524, 530, 531, 532, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 548, 599, 600, 610, 612, 670, 690, 694, 699, 700, 701, 702, 710, 715, 800, 900, 911, 912, 990, 999]),
        # from codebook
        'HIGRADED': valid_values_from_list([0, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 91, 92, 100, 101, 102, 110, 111, 112, 120, 121, 122, 130, 131, 132, 140, 141, 142, 150, 151, 152, 160, 161, 162, 170, 171, 172, 180, 181, 182, 190, 191, 192, 200, 201, 202, 210, 211, 212, 220, 221, 222, 230, 999]),
        # from codebook
        'EDUCD': valid_values_from_list([0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 30, 40, 50, 60, 61, 62, 63, 64, 65, 70, 71, 80, 81, 82, 83, 90, 100, 101, 110, 111, 112, 113, 114, 115, 116, 999]),
        # from codebook
        'OCC1950': valid_values_from_list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 123, 200, 201, 203, 204, 205, 210, 230, 240, 250, 260, 270, 280, 290, 300, 301, 302, 304, 305, 310, 320, 321, 322, 325, 335, 340, 341, 342, 350, 360, 365, 370, 380, 390, 400, 410, 420, 430, 450, 460, 470, 480, 490, 500, 501, 502, 503, 504, 505, 510, 511, 512, 513, 514, 515, 520, 521, 522, 523, 524, 525, 530, 531, 532, 533, 534, 535, 540, 541, 542, 543, 544, 545, 550, 551, 552, 553, 554, 555, 560, 561, 562, 563, 564, 565, 570, 571, 572, 573, 574, 575, 580, 581, 582, 583, 584, 585, 590, 591, 592, 593, 594, 595, 600, 601, 602, 603, 604, 605, 610, 611, 612, 613, 614, 615, 620, 621, 622, 623, 624, 625, 630, 631, 632, 633, 634, 635, 640, 641, 642, 643, 644, 645, 650, 660, 661, 662, 670, 671, 672, 673, 674, 675, 680, 681, 682, 683, 684, 685, 690, 700, 710, 720, 730, 731, 732, 740, 750, 751, 752, 753, 754, 760, 761, 762, 763, 764, 770, 771, 772, 773, 780, 781, 782, 783, 784, 785, 790, 810, 820, 830, 840, 910, 920, 930, 940, 950, 960, 970, 979, 980, 981, 982, 983, 984, 985, 986, 987, 990, 991, 995, 997, 999]),
        # from codebook
        'UOCC': valid_values_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 98, 99, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 156, 200, 210, 220, 222, 224, 226, 236, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 266, 270, 272, 274, 276, 278, 280, 282, 284, 286, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 327, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 260, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 496, 500, 510, 520, 600, 602, 604, 606, 608, 610, 612, 614, 700, 710, 712, 714, 720, 730, 732, 740, 750, 760, 770, 780, 790, 792, 794, 796, 798, 844, 866, 888, 900, 902, 904, 906, 908, 910, 988, 995, 996, 997, 998, 999]),
        # COUNTY is a geographic variable where we are allowed to pick the set of values from ground truth.
        # Coloardo counties from https://usa.ipums.org/usa/volii/ICPSR.shtml#colorado
        'COUNTY': valid_values_from_groundtruth(original_df, 'COUNTY'),

        'ENUMDIST': 'int',  #
        'CITYPOP': 'int',  # TODO(malzantot): fix
        'URBPOP': 'int',  # TODO(malzantot): fix
        # METAREAD is a geographic variable where we are allowed to pick the set of values from ground truth.
        'METAREAD': valid_values_from_groundtruth(original_df, 'METAREAD'),
        # from codebook
        'MTONGUED': valid_values_from_list([0, 100, 110, 120, 130, 140, 150, 160, 200, 210, 220, 230, 240, 300, 310, 320, 400, 410, 420, 430, 440, 450, 460, 470, 500, 600, 700, 800, 810, 900, 1000, 1010, 1020, 1030, 1100, 1110, 1120, 1130, 1140, 1150, 1200, 1210, 1220, 1230, 1240, 1250, 1300, 1400, 1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590, 1600, 1700, 1800, 1810, 1811, 1820, 1900, 1910, 1920, 1930, 2000, 2010, 2020, 2100, 2110, 2200, 2300, 2310, 2320, 2330, 2331, 2332, 2400, 2500, 2510, 2600, 2610, 2620, 2621, 2630, 2700, 2800, 2900, 2910, 3000, 3010, 3020, 3030, 3040, 3050, 3100, 3101, 3102, 3103, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3130, 3140, 3150, 3190, 3200, 3210, 3300, 3400, 3401, 3402, 3500, 3510, 3511, 3520, 3521, 3530, 3600, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3800, 3810, 3900, 4000, 4001, 4002, 4003, 4004, 4005, 4010, 4011, 4100, 4110, 4200, 4300, 4301, 4302, 4303, 4310, 4311, 4312, 4313, 4314, 4315, 4400, 4410, 4420, 4500, 4510, 4600, 4700, 4710, 4720, 4800, 4900, 5000, 5100, 5110, 5120, 5130, 5140, 5150, 5200, 5210, 5220, 5230, 5240, 5250, 5260, 5270, 5280, 5290, 5300, 5310, 5320, 5330, 5340, 5400, 5410, 5420, 5430, 5440, 5450, 5460, 5470, 5480, 5500, 5501, 5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5527, 5528, 5529, 5530, 5590, 5600, 5700, 5710, 5720, 5730, 5740, 5750, 5800, 5810, 5820, 5900, 6000, 6100, 6110, 6120, 6130, 6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312, 6313, 6314, 6320, 6321, 6322, 6390, 6400, 7000, 7100, 7110, 7120, 7130, 7140, 7150, 7160, 7200, 7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213, 7214, 7215, 7216, 7217, 7218, 7219, 7300, 7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311, 7312, 7313, 7314, 7400, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7410, 7411, 7412, 7413, 7420, 7421, 7422, 7423, 7424, 7430, 7440, 7450, 7490, 7500, 7600, 7610, 7620, 7630, 7700, 7701, 7702, 7703, 7704, 7705, 7706, 7707, 7708, 7709, 7710, 7711, 7712, 7713, 7714, 7715, 7800, 7900, 7910, 7920, 7930, 7940, 7950, 7960, 7970, 7980, 7990, 8000, 8010, 8020, 8030, 8040, 8050, 8060, 8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8120, 8200, 8210, 8220, 8230, 8240, 8250, 8260, 8300, 8400, 8410, 8420, 8430, 8440, 8450, 8460, 8470, 8480, 8500, 8510, 8520, 8530, 8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610, 8620, 8630, 8631, 8632, 8633, 8640, 8700, 8800, 8810, 8820, 8900, 8910, 9000, 9010, 9020, 9030, 9040, 9050, 9100, 9101, 9110, 9111, 9112, 9120, 9130, 9131, 9140, 9150, 9160, 9170, 9171, 9200, 9210, 9211, 9212, 9213, 9214, 9215, 9220, 9230, 9231, 9240, 9241, 9242, 9250, 9260, 9270, 9271, 9280, 9281, 9282, 9290, 9291, 9292, 9300, 9400, 9410, 9420, 9500, 9600, 9601, 9602, 9700, 9900, 9999]),
        'EDSCOR50': 'int_v',
        'MIGCITY5': 'int_v',
        'NPBOSS50': 'int_v',
        # CITY is a geographic variable where we are allowed to pick the set of values from ground truth.
        'CITY': valid_values_from_groundtruth(original_df, 'CITY'),
        'RENT': 'int_v',  # int or 9999
        'MIGCOUNTY': 'int_v',
        'MIGMET5': 'int_v',
        'ERSCOR50': 'int_v',
        'MBPLD': 'void',  # This one will be equal MBPL * 100
        'FBPLD': 'void',  # This one will be equal to FBPL * 100
        'BPLD': 'void',  # this one will be equal to BPL * 100
        'INCWAGE': 'int_v',
        'VALUEH': 'int_v'
    }
    #select_cols = ['SPLIT', 'OWNERSHP', 'VETWWI','VALUEH', 'CITYPOP']
    #original_df = original_df[select_cols]
    #col_maps = {k:col_maps[k] for k in select_cols}
    output_df_columns = []
    for k in columns_list:
        assert k in columns_list, 'Cannot find pre-prepossing of column {}'.format(k)
        v = col_maps[k]
        if isinstance(v, dict):
            print('Processing [ {} ] '.format(k))
            mapped_col = original_df[k].map(v)
            if (len(v) == 2):
                output_df_columns.append(mapped_col)
            else:
                mapped_vals = mapped_col.values.reshape((-1, 1))
                ohe_col = pd.DataFrame(
                    data=OneHotEncoder(n_values=len(v), sparse=False).fit_transform(mapped_vals).astype(np.float32), index=original_df.index)
                output_df_columns.append(ohe_col)
        elif v == 'void':
            pass  # skip that column
        elif v == 'int':
            output_df_columns.append(original_df[k] / 100.0)
        elif v == 'int_v':
            val_column = original_df[k].values.astype(np.float32)
            is_valid_column = (
                original_df[k] != metadata[k]['maxval']).astype(np.float32)
            print(k, ' - ', is_valid_column.sum(),
                  ' / ', is_valid_column.shape[0])
            val_column_processed = (is_valid_column) * val_column / 100.0
            output_df_columns.append(
                is_valid_column.rename('{}_valid'.format(k)))
            output_df_columns.append(val_column_processed)
        else:
            raise Exception('Invalid mapping')
    output_df = pd.concat(output_df_columns, 1)
    print(output_df.shape)
    return original_df, output_df, metadata, col_maps, columns_list


def postprocess_data(input_data, metadata, col_maps, columns_list, greedy=True):
    """ Applies post-processing to the generator model outputs """
    output_df_columns = {}
    cur_idx = 0
    if isinstance(input_data, np.ndarray):
        input_data = pd.DataFrame(data=input_data)
    for k in columns_list:
        assert k in col_maps, 'Coloumn mapping not found'
        v = col_maps[k]
        print('Post processing {}'.format(k))
        if isinstance(v, dict):
            col_start = cur_idx
            if len(v) == 2:
                # binary column
                col_end = col_start + 1
                output_col = (
                    input_data.iloc[:, col_start] > 0.5).astype(np.int32)
                cur_idx += 1
            else:
                if greedy:
                    col_end = col_start + len(v)
                    col_ohe = input_data.iloc[:, col_start: col_end]
                    output_col = pd.Series(
                        data=np.argmax(
                            col_ohe.values, axis=1).astype(np.int32),
                        index=input_data.index)

                    cur_idx += len(v)
                else:
                    col_end = col_start + 1
                    output_col = input_data.iloc[:, col_start].astype(np.int32)
                    cur_idx += 1
            inv_map = {cv: ck for ck, cv in v.items()}
            output_col = output_col.map(inv_map)
            output_df_columns[k] = output_col.astype(np.int32)
        elif v == 'int':
            val_col = (100.0 * input_data.iloc[:, cur_idx]).astype(np.int32)
            val_col = np.clip(val_col, 0, metadata[k]['maxval'])
            output_df_columns[k] = val_col
            cur_idx += 1
        elif v == 'int_v':
            is_valid_column = (input_data.iloc[:, cur_idx] > 0.5)
            value_column = (
                100.0 * input_data.iloc[:, cur_idx+1]).astype(np.int32)
            value_column = np.clip(value_column, 0, metadata[k]['maxval'])
            val_column_processed = (
                is_valid_column * value_column) + (1-is_valid_column) * metadata[k]['maxval']
            output_df_columns[k] = val_column_processed.astype(np.int32)
            cur_idx += 2
        elif v == 'void':
            if k == 'MBPLD':
                output_df_columns[k] = (
                    output_df_columns['MBPL'].values.reshape((-1,)) * 100).astype(np.int32)
            elif k == 'FBPLD':
                output_df_columns[k] = (
                    output_df_columns['FBPL'].values.reshape(-1,) * 100).astype(np.int32)
            elif k == 'BPLD':
                output_df_columns[k] = (
                    output_df_columns['BPL'].values.reshape((-1,)) * 100).astype(np.int32)
            else:
                raise Exception('Invalid mapping for column {}'.format(k))
        else:
            raise Exception("Invalid mapping for column {}". format(k))
    output_df = pd.DataFrame(output_df_columns)
    return output_df



if __name__ == '__main__':
    """ Only for testing """
    original_df, output_data, metadata, col_maps, columns_list = preprocess_nist_data(
        '../data/colorado.csv',
        '../data/colorado-specs.json', subsample=True)
    output_df = postprocess_data(output_data, metadata, col_maps, columns_list, greedy=True)
    assert(output_df.shape == original_df.shape)
    match_count = 0
    for i in range(output_df.shape[0]):
        if pd.DataFrame.equals(output_df.iloc[i, :], original_df.iloc[i, :]):
            match_count += 1
    print('Match ratio : {}/ {}'.format(match_count, output_df.shape[0]))
