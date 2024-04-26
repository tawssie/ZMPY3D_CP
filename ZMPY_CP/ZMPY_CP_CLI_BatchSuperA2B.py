# MIT License
#
# Copyright (c) 2024 Jhih-Siang Lai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import cupy as cp
import pickle
import argparse
import os
import sys

import ZMPY_CP as z

def ZMPY_CP_CLI_BatchSuperA2B(PDBFileNameA, PDBFileNameB):
    def ZMCal(PDBFileName,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear, MaxOrder, Param, ResidueBox, RotationIndex):

        [XYZ,AA_NameList]=z.get_pdb_xyz_ca(PDBFileName)

        [Voxel3D,Corner]=z.fill_voxel_by_weight_density(XYZ,AA_NameList,Param['residue_weight_map'],GridWidth,ResidueBox[GridWidth])
        Voxel3D=cp.array(Voxel3D,dtype=cp.float64)
        Corner=cp.array(Corner,dtype=cp.float64)
        
        Dimension_BBox_scaled=cp.shape(Voxel3D)
        Dimension_BBox_scaled=cp.array(Dimension_BBox_scaled,dtype=cp.int32)
        GridWidth=cp.array(GridWidth,dtype=cp.float64)
        
        MaxOrder=cp.array(MaxOrder,dtype=cp.int64)
        
        X_sample = cp.arange(Dimension_BBox_scaled[0] + 1, dtype=cp.float64)
        Y_sample = cp.arange(Dimension_BBox_scaled[1] + 1, dtype=cp.float64)
        Z_sample = cp.arange(Dimension_BBox_scaled[2] + 1, dtype=cp.float64)
    
        [VolumeMass,Center,_]=z.calculate_bbox_moment(Voxel3D,1,X_sample,Y_sample,Z_sample)
    
        [AverageVoxelDist2Center,MaxVoxelDist2Center]=z.calculate_molecular_radius(Voxel3D,Center,VolumeMass,cp.array(Param['default_radius_multiplier'], dtype=cp.float64))        

        ##################################################################################
        # You may add any preprocessing on the voxel before applying the Zernike moment. #
        ##################################################################################
        
        Center_scaled=Center*GridWidth+Corner
        
        Sphere_X_sample, Sphere_Y_sample, Sphere_Z_sample=z.get_bbox_moment_xyz_sample(Center,AverageVoxelDist2Center,Dimension_BBox_scaled)
    
        _,_,SphereBBoxMoment=z.calculate_bbox_moment(Voxel3D
                                          ,MaxOrder
                                          ,Sphere_X_sample
                                          ,Sphere_Y_sample
                                          ,Sphere_Z_sample)
        
        ZMoment_scaled,ZMoment_raw=z.calculate_bbox_moment_2_zm(MaxOrder
                                           , GCache_complex
                                           , GCache_pqr_linear
                                           , GCache_complex_index
                                           , CLMCache3D
                                           , SphereBBoxMoment)        

        ABList_2=z.calculate_ab_rotation_all(ZMoment_raw.get(), 2)
        ABList_3=z.calculate_ab_rotation_all(ZMoment_raw.get(), 3)
        ABList_4=z.calculate_ab_rotation_all(ZMoment_raw.get(), 4)
        ABList_5=z.calculate_ab_rotation_all(ZMoment_raw.get(), 5)
        ABList_6=z.calculate_ab_rotation_all(ZMoment_raw.get(), 6)

        ABList_all=cp.vstack(ABList_2+ABList_3+ABList_4+ABList_5+ABList_6)

        ZMList_all=z.calculate_zm_by_ab_rotation(ZMoment_raw, BinomialCache, ABList_all, MaxOrder, CLMCache,s_id,n,l,m,mu,k,IsNLM_Value)

        ZMList_all=cp.transpose(ZMList_all,(2,1,0,3)) 
        ZMList_all=ZMList_all[~cp.isnan(ZMList_all)]
        # Based on ABList_all, it is known in advance that Order 6 will definitely have 96 pairs of AB, which means 96 vectors.
        ZMList_all=cp.reshape(ZMList_all,(cp.int64(ZMList_all.size/96),96))

        return XYZ, Center_scaled, ABList_all,ZMList_all,AA_NameList

    Param=z.get_global_parameter();

    MaxOrder=6

    BinomialCacheFilePath = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_data'), 'BinomialCache.pkl')
    with open(BinomialCacheFilePath, 'rb') as file: # Used at the entry point, it requires __file__ to identify the package location
    # with open('./cache_data/BinomialCache.pkl', 'rb') as file: # Can be used in ipynb, but not at the entry point. 
        BinomialCachePKL = pickle.load(file)

    LogCacheFilePath=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_data'), 'LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder))
    with open(LogCacheFilePath, 'rb') as file: # Used at the entry point, it requires __file__ to identify the package location
    # with open('./cache_data/LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder), 'rb') as file: # Can be used in ipynb, but not at the entry point. 
        CachePKL = pickle.load(file)  

    # Extract all cached variables from pickle. These will be converted into a tensor/cupy objects for ZMPY_CP and ZMPY_TF.
    BinomialCache= cp.array(BinomialCachePKL['BinomialCache'],dtype=cp.float64)
    
    # GCache, CLMCache, and all RotationIndex
    GCache_pqr_linear= cp.array(CachePKL['GCache_pqr_linear'], dtype=cp.int32)
    GCache_complex= cp.array(CachePKL['GCache_complex'],dtype=cp.complex128)
    GCache_complex_index= cp.array(CachePKL['GCache_complex_index'], dtype=cp.int32)
    CLMCache3D= cp.array(CachePKL['CLMCache3D'],dtype=cp.complex128)
    CLMCache= cp.array(CachePKL['CLMCache'], dtype=cp.float64)
    

    RotationIndex=CachePKL['RotationIndex']

    # RotationIndex is a structure, must be [0,0] to accurately obtain the s_id ... etc, within RotationIndex.
    s_id=cp.array(np.squeeze(RotationIndex['s_id'][0,0])-1, dtype=cp.int32)
    n   =cp.array(np.squeeze(RotationIndex['n'] [0,0]), dtype=cp.int32)
    l   =cp.array(np.squeeze(RotationIndex['l'] [0,0]), dtype=cp.int32)
    m   =cp.array(np.squeeze(RotationIndex['m'] [0,0]), dtype=cp.int32)
    mu  =cp.array(np.squeeze(RotationIndex['mu'][0,0]), dtype=cp.int32)
    k   =cp.array(np.squeeze(RotationIndex['k'] [0,0]), dtype=cp.int32)
    IsNLM_Value=cp.array(np.squeeze(RotationIndex['IsNLM_Value'][0,0])-1, dtype=cp.int32)  
    
    ResidueBox=z.get_residue_gaussian_density_cache(Param)

    GridWidth= 1.00; 

    TargetRotM = []
    
    for index in range(len(PDBFileNameA)):
        _, Center_scaled_A,ABList_A,ZMList_A,_ =ZMCal(PDBFileNameA[index],1.00,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear, MaxOrder, Param, ResidueBox, RotationIndex)
        _, Center_scaled_B,ABList_B,ZMList_B,_ =ZMCal(PDBFileNameB[index],1.00,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear, MaxOrder, Param, ResidueBox, RotationIndex)

        M = cp.abs(ZMList_A.conj().T @ ZMList_B) # square matrix A^T*B 
        MaxValueIndex = cp.where(M == cp.max(M)) # MaxValueIndex is a tuple that contains an nd array.
    
        i, j = MaxValueIndex[0][0], MaxValueIndex[1][0]
        
        RotM_A=z.get_transform_matrix_from_ab_list(ABList_A[i,0],ABList_A[i,1],Center_scaled_A)
        RotM_B=z.get_transform_matrix_from_ab_list(ABList_B[j,0],ABList_B[j,1],Center_scaled_B)

        TargetRotM.append(cp.linalg.solve(RotM_B, RotM_A))

    return TargetRotM



def main():
    if len(sys.argv) != 2:
        print('Usage: ZMPY_CP_CLI_BatchSuperA2B PDBFileList.txt')
        print('       This function takes a list of paired PDB structure file paths to generate transformation matrices.')
        print("Error: You must provide exactly one input file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Process input file that contains paths to .pdb or .txt files.')
    parser.add_argument('input_file', type=str, help='The input file that contains paths to .pdb or .txt files.')

    args = parser.parse_args()

    input_file = args.input_file
    if not input_file.endswith('.txt'):
        parser.error("File must end with .txt")
    
    if not os.path.isfile(input_file):
        parser.error("File does not exist")

    with open(input_file, 'r') as file:
        lines = file.readlines()

    file_list_1 = []
    file_list_2 = []
    for line in lines:

        files = line.strip().split()
        if len(files) != 2:
            print(f"Error: Each line must contain exactly two file paths, but got {len(files)}.")
            sys.exit(1)
        file1, file2 = files

        for file in [file1, file2]:
            if not (file.endswith('.pdb') or file.endswith('.txt')):
                print(f"Error: File {file} must end with .pdb or .txt.")
                sys.exit(1)
            if not os.path.isfile(file):
                print(f"Error: File {file} does not exist.")
                sys.exit(1)
        file_list_1.append(file1)
        file_list_2.append(file2)

    TargetRotM=ZMPY_CP_CLI_BatchSuperA2B(file_list_1, file_list_2)

    for M in TargetRotM:
        print(M)

if __name__ == '__main__':
    main()