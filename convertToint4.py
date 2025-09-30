import numpy as np
import sys


if len(sys.argv) <= 1:
    print("input file needed! Try: \"python convertToint4.py yourFile.npy\"")
    exit()

# load data
npy_load = np.load(sys.argv[1])
npy_8bit = npy_load.astype(np.ubyte)

# output data must be double the size of the input layer because of uint4
output_data = np.zeros((1,3980), dtype=np.ubyte)


output_data_index = 0

# convert a byte into two nibble
# 197   ->    1100 0101  ->  12      5
for i in npy_8bit[0]:
    byte_high = np.right_shift(i, 4)
    byte_low = np.bitwise_and(i, 0x0F)
    
    output_data[0][output_data_index] = byte_high
    output_data[0][output_data_index+1] = byte_low
    
    output_data_index+=2

# save output
np.save("output.npy",output_data, allow_pickle=False)


 