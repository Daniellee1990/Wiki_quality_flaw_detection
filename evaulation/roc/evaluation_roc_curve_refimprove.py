# coding='UTF-8'
import io
import sys
import os
import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
import tensorflow
import keras
from keras import regularizers, optimizers
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional, Flatten, Dropout, Multiply, Permute, concatenate
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc

from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # roc values get from the training model

    ### Refimprove  FPR, TPR, auc
    # content+stats bigru
    content_stats_bigru = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021598272138228943, 0.0021598272138228943, 0.004319654427645789, 0.004319654427645789, 0.0064794816414686825, 0.0064794816414686825, 0.008639308855291577, 0.008639308855291577, 0.01079913606911447, 0.01079913606911447, 0.012958963282937365, 0.012958963282937365, 0.01511879049676026, 0.01511879049676026, 0.017278617710583154, 0.017278617710583154, 0.02159827213822894, 0.02159827213822894, 0.02591792656587473, 0.02591792656587473, 0.028077753779697623, 0.028077753779697623, 0.03023758099352052, 0.03023758099352052, 0.04319654427645788, 0.04319654427645788, 0.04967602591792657, 0.04967602591792657, 0.06479481641468683, 0.06479481641468683, 0.10367170626349892, 0.10367170626349892, 0.13174946004319654, 0.13174946004319654, 0.17926565874730022, 0.17926565874730022, 1.0],
                           [0.0, 0.004634994206257242, 0.03360370799536501, 0.05909617612977984, 0.07995365005793743, 0.09617612977983778, 0.11123986095017381, 0.12166859791425261, 0.13904982618771727, 0.14831981460023175, 0.15990730011587487, 0.1645422943221321, 0.17960602549246812, 0.186558516801854, 0.19235225955967555, 0.1993047508690614, 0.20625724217844726, 0.21320973348783315, 0.21668597914252608, 0.22247972190034762, 0.22479721900347624, 0.22943221320973348, 0.23522595596755505, 0.2398609501738123, 0.24333719582850522, 0.24913093858632676, 0.25840092699884126, 0.2607184241019699, 0.2630359212050985, 0.2688296639629201, 0.272305909617613, 0.2769409038238702, 0.28157589803012745, 0.2850521436848204, 0.29200463499420626, 0.30127462340672073, 0.30359212050984935, 0.3174971031286211, 0.32444959443800697, 0.32560834298957125, 0.3290845886442642, 0.3314020857473928, 0.33603707995365006, 0.3406720741599073, 0.34414831981460026, 0.3464658169177289, 0.3499420625724218, 0.3522595596755504, 0.35341830822711473, 0.35689455388180763, 0.35921205098493625, 0.3603707995365006, 0.3650057937427578, 0.36732329084588644, 0.36964078794901506, 0.373117033603708, 0.37543453070683663, 0.37891077636152953, 0.38122827346465815, 0.3858632676709154, 0.3870220162224797, 0.3904982618771727, 0.39165701042873696, 0.3962920046349942, 0.40092699884125144, 0.4020857473928158, 0.4044032444959444, 0.4055619930475087, 0.40903823870220163, 0.4101969872537659, 0.41367323290845887, 0.4159907300115875, 0.42526071842410196, 0.4322132097334878, 0.43453070683661643, 0.43568945538818077, 0.4380069524913094, 0.4426419466975666, 0.4461181923522596, 0.4484356894553882, 0.4495944380069525, 0.45307068366164543, 0.45654692931633833, 0.46349942062572425, 0.4646581691772885, 0.46697566628041715, 0.4681344148319815, 0.4727694090382387, 0.4762456546929316, 0.47856315179606024, 0.4820393974507532, 0.48899188876013905, 0.49130938586326767, 0.4947856315179606, 0.49710312862108924, 0.49942062572421786, 0.5040556199304751, 0.507531865585168, 0.5110081112398609, 0.5144843568945539, 0.5179606025492468, 0.522595596755504, 0.5237543453070683, 0.5283893395133256, 0.5318655851680185, 0.5365005793742758, 0.5376593279258401, 0.5422943221320974, 0.544611819235226, 0.5469293163383546, 0.5492468134414832, 0.5515643105446119, 0.5596755504055619, 0.5619930475086906, 0.5701042873696408, 0.5724217844727694, 0.5955967555040557, 0.5979142526071842, 0.6129779837775203, 0.6152954808806489, 0.6210892236384704, 0.6234067207415991, 0.6778679026651216, 0.6825028968713789, 0.7114716106604867, 0.7137891077636153, 0.7288528389339514, 0.7334878331402086, 0.9606025492468134, 0.9606025492468134, 0.9617612977983777, 0.9617612977983777, 0.9721900347624566, 0.9721900347624566, 0.9745075318655851, 0.9745075318655851, 0.9756662804171495, 0.9756662804171495, 0.9768250289687138, 0.9768250289687138, 0.9779837775202781, 0.9779837775202781, 0.9791425260718424, 0.9791425260718424, 0.9803012746234068, 0.9803012746234068, 0.9837775202780996, 0.9837775202780996, 0.9860950173812283, 0.9860950173812283, 0.9872537659327926, 0.9872537659327926, 0.9907300115874855, 0.9907300115874855, 0.9930475086906141, 0.9930475086906141, 0.9942062572421785, 0.9942062572421785, 0.9953650057937428, 0.9953650057937428, 0.996523754345307, 0.996523754345307, 0.9976825028968713, 0.9976825028968713, 1.0, 1.0],
                           [0.998668565379196]]

    # content bigru
    content_bigru = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.0021598272138228943, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.0064794816414686825, 0.0064794816414686825, 0.008639308855291577, 0.008639308855291577, 0.01079913606911447, 0.01079913606911447, 0.012958963282937365, 0.012958963282937365, 0.017278617710583154, 0.017278617710583154, 0.019438444924406047, 0.019438444924406047, 0.023758099352051837, 0.023758099352051837, 0.04103671706263499, 0.04103671706263499, 0.04319654427645788, 0.04319654427645788, 0.06047516198704104, 0.06047516198704104, 0.06695464362850972, 0.06695464362850972, 0.0755939524838013, 0.0755939524838013, 0.08639308855291576, 0.08639308855291576, 0.10367170626349892, 0.10367170626349892, 0.11879049676025918, 0.11879049676025918, 0.12742980561555076, 0.12742980561555076, 0.12958963282937366, 0.12958963282937366, 0.1468682505399568, 0.1468682505399568, 0.15334773218142547, 0.15334773218142547, 0.16630669546436286, 0.16630669546436286, 0.22894168466522677, 0.22894168466522677, 0.28293736501079914, 0.28293736501079914, 0.28725701943844495, 0.28725701943844495, 0.4319654427645788, 0.4319654427645788, 0.46004319654427644, 0.46004319654427644, 0.4946004319654428, 0.4946004319654428, 0.7580993520518359, 0.7580993520518359, 0.9892008639308856, 0.9892008639308856, 1.0],
                     [0.0, 0.0011587485515643105, 0.010428736964078795, 0.013904982618771726, 0.023174971031286212, 0.02549246813441483, 0.047508690614136734, 0.04982618771726535, 0.07184241019698726, 0.07415990730011587, 0.07647740440324449, 0.07879490150637311, 0.09733487833140209, 0.0996523754345307, 0.10660486674391657, 0.10892236384704519, 0.1100811123986095, 0.11239860950173812, 0.11587485515643106, 0.11819235225955968, 0.12514484356894554, 0.12862108922363846, 0.13093858632676708, 0.13441483198146004, 0.13789107763615296, 0.1413673232908459, 0.14484356894553882, 0.14716106604866744, 0.14947856315179606, 0.15063731170336037, 0.15295480880648898, 0.1552723059096176, 0.15758980301274622, 0.16106604866743918, 0.1633835457705678, 0.16801853997682503, 0.17265353418308227, 0.18076477404403243, 0.1853997682502897, 0.186558516801854, 0.19119351100811124, 0.19351100811123986, 0.19582850521436848, 0.1981460023174971, 0.20509849362688296, 0.21089223638470453, 0.21320973348783315, 0.21668597914252608, 0.2190034762456547, 0.22247972190034762, 0.22711471610660486, 0.22943221320973348, 0.23174971031286212, 0.23754345307068367, 0.2398609501738123, 0.2410196987253766, 0.24333719582850522, 0.25028968713789107, 0.253765932792584, 0.25840092699884126, 0.2607184241019699, 0.2630359212050985, 0.2653534183082271, 0.26998841251448435, 0.2734646581691773, 0.27578215527230593, 0.28041714947856317, 0.2827346465816918, 0.2850521436848204, 0.2862108922363847, 0.28968713789107764, 0.2908458864426419, 0.2931633835457706, 0.29779837775202783, 0.30011587485515645, 0.30127462340672073, 0.30590961761297797, 0.3082271147161066, 0.31170336037079954, 0.31402085747392816, 0.3163383545770568, 0.3186558516801854, 0.32213209733487835, 0.32329084588644263, 0.32560834298957125, 0.3267670915411356, 0.3290845886442642, 0.3325608342989571, 0.33603707995365006, 0.33719582850521435, 0.33951332560834296, 0.3429895712630359, 0.34530706836616454, 0.34762456546929316, 0.3499420625724218, 0.35341830822711473, 0.35689455388180763, 0.35921205098493625, 0.36152954808806487, 0.3638470451911935, 0.3684820393974508, 0.3719582850521437, 0.3742757821552723, 0.3765932792584009, 0.37891077636152953, 0.388180764774044, 0.3904982618771727, 0.3928157589803013, 0.3951332560834299, 0.39976825028968715, 0.4020857473928158, 0.40324449594438005, 0.406720741599073, 0.41135573580533025, 0.4183082271147161, 0.4241019698725377, 0.42526071842410196, 0.4298957126303592, 0.4322132097334878, 0.43568945538818077, 0.4380069524913094, 0.4519119351100811, 0.4542294322132097, 0.45770567786790267, 0.4600231749710313, 0.46349942062572425, 0.46581691772885286, 0.4704519119351101, 0.47740440324449596, 0.4797219003476246, 0.48088064889918886, 0.4843568945538818, 0.48667439165701043, 0.48899188876013905, 0.49130938586326767, 0.4936268829663963, 0.4959443800695249, 0.5121668597914253, 0.5144843568945539, 0.5191193511008111, 0.5214368482039398, 0.522595596755504, 0.5249130938586327, 0.5376593279258401, 0.5399768250289687, 0.541135573580533, 0.5457705677867902, 0.5480880648899189, 0.5504055619930475, 0.5585168018539977, 0.5608342989571263, 0.5689455388180765, 0.5712630359212051, 0.5874855156431055, 0.589803012746234, 0.589803012746234, 0.5909617612977984, 0.593279258400927, 0.6048667439165701, 0.6071842410196987, 0.6199304750869061, 0.6222479721900348, 0.626882966396292, 0.6292004634994206, 0.6361529548088065, 0.6384704519119351, 0.6488991888760139, 0.6512166859791425, 0.6523754345307068, 0.657010428736964, 0.6662804171494786, 0.6697566628041715, 0.6743916570104287, 0.6767091541135574, 0.6825028968713789, 0.6848203939745076, 0.6882966396292005, 0.6906141367323291, 0.7045191193511008, 0.7068366164542295, 0.7079953650057937, 0.7103128621089224, 0.7415990730011588, 0.7439165701042874, 0.7555040556199305, 0.7578215527230591, 0.8006952491309386, 0.8030127462340672, 0.8111239860950173, 0.813441483198146, 0.8157589803012746, 0.8180764774044033, 0.9038238702201622, 0.9061413673232909, 0.9119351100811124, 0.9142526071842411, 0.93279258400927, 0.93279258400927, 0.9351100811123986, 0.9351100811123986, 0.9536500579374276, 0.9536500579374276, 0.9606025492468134, 0.9606025492468134, 0.9629200463499421, 0.9629200463499421, 0.9675550405561993, 0.9675550405561993, 0.9710312862108922, 0.9710312862108922, 0.9733487833140209, 0.9733487833140209, 0.9756662804171495, 0.9756662804171495, 0.9768250289687138, 0.9768250289687138, 0.9779837775202781, 0.9779837775202781, 0.9791425260718424, 0.9791425260718424, 0.9803012746234068, 0.9803012746234068, 0.981460023174971, 0.981460023174971, 0.9826187717265353, 0.9826187717265353, 0.984936268829664, 0.984936268829664, 0.9860950173812283, 0.9860950173812283, 0.9872537659327926, 0.9872537659327926, 0.9884125144843569, 0.9884125144843569, 0.9895712630359212, 0.9895712630359212, 0.9907300115874855, 0.9907300115874855, 0.9918887601390498, 0.9918887601390498, 0.9930475086906141, 0.9930475086906141, 0.9942062572421785, 0.9942062572421785, 0.9953650057937428, 0.9953650057937428, 0.996523754345307, 0.996523754345307, 0.9976825028968713, 0.9976825028968713, 0.9988412514484357, 0.9988412514484357, 1.0, 1.0],
                     [0.9914069910328379]]

    # stats dnn
    stats_dnn = [[0.0, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.004319654427645789, 0.0064794816414686825, 0.0064794816414686825, 0.008639308855291577, 0.008639308855291577, 0.012958963282937365, 0.012958963282937365, 0.019438444924406047, 0.019438444924406047, 0.028077753779697623, 0.028077753779697623, 0.03023758099352052, 0.03023758099352052, 0.032397408207343416, 0.032397408207343416, 0.03455723542116631, 0.03455723542116631, 0.04103671706263499, 0.04103671706263499, 0.04967602591792657, 0.04967602591792657, 0.06263498920086392, 0.06263498920086392, 0.06479481641468683, 0.06479481641468683, 0.09503239740820735, 0.09503239740820735, 0.11231101511879049, 0.11231101511879049, 0.12958963282937366, 0.12958963282937366, 0.17278617710583152, 0.17278617710583152, 0.3282937365010799, 0.3282937365010799, 0.5032397408207343, 0.5032397408207343, 0.6436285097192225, 0.6436285097192225, 1.0],
                 [0.0, 0.40324449594438005, 0.4646581691772885, 0.5052143684820394, 0.5341830822711472, 0.5480880648899189, 0.5596755504055619, 0.5724217844727694, 0.5863267670915412, 0.5979142526071842, 0.6118192352259559, 0.6199304750869061, 0.6257242178447276, 0.6326767091541136, 0.6407879490150638, 0.645422943221321, 0.6512166859791425, 0.660486674391657, 0.6674391657010429, 0.6767091541135574, 0.6813441483198146, 0.6836616454229433, 0.6848203939745076, 0.6882966396292005, 0.6906141367323291, 0.6952491309385863, 0.6987253765932793, 0.6998841251448435, 0.7022016222479722, 0.7056778679026651, 0.7079953650057937, 0.709154113557358, 0.7114716106604867, 0.7161066048667439, 0.7184241019698725, 0.7334878331402086, 0.7358053302433372, 0.7392815758980301, 0.7415990730011588, 0.7520278099652375, 0.7543453070683661, 0.7705677867902665, 0.7728852838933952, 0.7740440324449595, 0.776361529548088, 0.7902665121668598, 0.7925840092699884, 0.8250289687137891, 0.8273464658169177, 0.8829663962920047, 0.8852838933951332, 0.9536500579374276, 0.9536500579374276, 0.9698725376593279, 0.9698725376593279, 0.9733487833140209, 0.9733487833140209, 0.9768250289687138, 0.9768250289687138, 0.9779837775202781, 0.9779837775202781, 0.981460023174971, 0.981460023174971, 0.9826187717265353, 0.9826187717265353, 0.9837775202780996, 0.9837775202780996, 0.9860950173812283, 0.9860950173812283, 0.9872537659327926, 0.9872537659327926, 0.9895712630359212, 0.9895712630359212, 0.9907300115874855, 0.9907300115874855, 0.9918887601390498, 0.9918887601390498, 0.9930475086906141, 0.9930475086906141, 0.9942062572421785, 0.9942062572421785, 0.9953650057937428, 0.9953650057937428, 0.996523754345307, 0.996523754345307, 0.9976825028968713, 0.9976825028968713, 0.9988412514484357, 0.9988412514484357, 1.0, 1.0],
                 [0.9936881990344596]]


    content_stats_bigru_fpr = np.array(content_stats_bigru[0])
    content_stats_bigru_tpr = np.array(content_stats_bigru[1])
    content_stats_bigru_roc_auc = content_stats_bigru[2][0]

    content_bigru_fpr = np.array(content_bigru[0])
    content_bigru_tpr = np.array(content_bigru[1])
    content_bigru_roc_auc = content_bigru[2][0]

    stats_dnn_fpr = np.array(stats_dnn[0])
    stats_dnn_tpr = np.array(stats_dnn[1])
    stats_dnn_roc_auc = stats_dnn[2][0]


    #开始画ROC曲线
    lw = 1.5
    plt.plot(content_stats_bigru_fpr, content_stats_bigru_tpr, 
            color='#FF8C00', linestyle='-', lw=lw, label='Content+stats BiGRU (AUC = %0.4f)'% content_stats_bigru_roc_auc)
    plt.plot(content_bigru_fpr, content_bigru_tpr, 
            color='#A52A2A', linestyle='--', lw=lw,label='Content BiGRU (AUC = %0.4f)'% content_bigru_roc_auc)
    plt.plot(stats_dnn_fpr, stats_dnn_tpr, 
            color='#ADFF2F', linestyle='-', lw=lw,label='Stats DNN (AUC = %0.4f)'% stats_dnn_roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],color='#191970', linestyle='--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic')
    plt.show()