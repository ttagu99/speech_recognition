import numpy as np
import pandas as pd
import os


root_folder = 'I:/imgfolder/voice/'
input_csv1 = '1dcnn_last32_noiseadd_ts_mul_balance_inputnormal_submean_abs_whitenadd_sgd.hdf5_prob_.csv' #0.85
input_csv2 = 'resnet_addlayer_last512_noiseadd_ts_mul_balance_inputnormal_submean_abs_whitenadd.hdf5_prob.csv' #0.85
input_csv3 = '1dcnn_last1024_noiseadd_ts_mul_balance_inputnormal_submean_abs_whitenadd_sgd.hdf5_prob.csv' #0.86
input_csv4 = 'combi_1dvgg_2dres.hdf5_prob.csv' #0.85
synsets = 'yes no up down left right on off stop go silence unknown'.split()
outFileName = 'ensemble_2_1d_1024_2dres.csv'
input1 = open(root_folder + input_csv1, 'r')
input2 = open(root_folder + input_csv2, 'r')
input3 = open(root_folder +input_csv3,'r')
input4 = open(root_folder +input_csv4,'r')

outFile = open(root_folder + outFileName, 'w')



num_products = 158538

outFile.write('fname,label\n')
# header read
line_input1 = input1.readline()
line_input2 = input2.readline()
line_input3 = input3.readline()
line_input4 = input4.readline()
# data read
for x in range(num_products):
    line_input1 = input1.readline()
    line_input2 = input2.readline()
    line_input3 = input3.readline()
    line_input4 = input4.readline()

    pars1 = line_input1.split(',')
    pars2 = line_input2.split(',')
    pars3 = line_input3.split(',')
    pars4 = line_input4.split(',')

    prob1 = np.array([float(value) for value in pars1[1:]])
    prob2 = np.array([float(value) for value in pars2[1:]])
    prob3 = np.array([float(value) for value in pars3[1:]])
    prob4 = np.array([float(value) for value in pars4[1:]])
    sumprob =prob2+ prob3 # prob1 + prob2 + prob3 + prob4
    max_index = sumprob.argmax()
    max_value = sumprob.max()

    outstr = pars1[0] + ',' + synsets[max_index] + '\n'
    outFile.write(outstr)

    if x % 1000 == 0 :
        logstr = str(x)  + 'rows'
        print (logstr)

    if pars1[0] != pars2[0] or pars2[0] != pars3[0]  or pars3[0] != pars4[0] :
        logstr = 'warning not match row image name'
        print(logstr)

