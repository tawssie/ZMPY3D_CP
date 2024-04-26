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

def ZMPY_CP_CLI_ShapeScore(PDBFileNameA, PDBFileNameB,GridWidth):
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

        ZMList=[]

        # ZMoment_scaled[np.isnan(ZMoment_raw)]=np.nan
        ZM_3DZD_invariant=z.get_3dzd_121_descriptor(ZMoment_scaled)

        ZMList.append(ZM_3DZD_invariant)

        MaxTargetOrder2NormRotate=5
        for TargetOrder2NormRotate in range(2, MaxTargetOrder2NormRotate+1):
            ABList = z.calculate_ab_rotation(ZMoment_raw.get(), TargetOrder2NormRotate)  
            ABList = cp.array(ABList)
            ZM = z.calculate_zm_by_ab_rotation(ZMoment_raw, BinomialCache, ABList, MaxOrder, CLMCache,s_id,n,l,m,mu,k,IsNLM_Value)
            ZM_mean, _ = z.get_mean_invariant(ZM)
            ZMList.append(ZM_mean)

        MomentInvariant = cp.concatenate([z[~cp.isnan(z)] for z in ZMList])

        TotalResidueWeight=z.get_total_residue_weight(AA_NameList,Param['residue_weight_map'])

        [Prctile_list,STD_XYZ_dist2center,S,K]=z.get_ca_distance_info(cp.asarray(XYZ))

        GeoDescriptor = cp.vstack((AverageVoxelDist2Center, TotalResidueWeight, Prctile_list, STD_XYZ_dist2center, S, K))
        
        return MomentInvariant, GeoDescriptor
    
    Param=z.get_global_parameter();

    MaxOrder=20
    
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

    MaxN=MaxOrder+1
    
    ResidueBox=z.get_residue_gaussian_density_cache(Param)

    MomentInvariantRawA, GeoDescriptorA =ZMCal(PDBFileNameA,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear, MaxOrder, Param, ResidueBox, RotationIndex)
    MomentInvariantRawB, GeoDescriptorB =ZMCal(PDBFileNameB,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear, MaxOrder, Param, ResidueBox, RotationIndex)

    P=z.get_descriptor_property()

    ZMIndex  = cp.vstack([P['ZMIndex0'], P['ZMIndex1'], P['ZMIndex2'], P['ZMIndex3'], P['ZMIndex4']])
    ZMWeight = cp.vstack([P['ZMWeight0'], P['ZMWeight1'], P['ZMWeight2'], P['ZMWeight3'], P['ZMWeight4']])
    
    # Calculating ZMScore
    ZMScore = cp.sum(cp.abs(MomentInvariantRawA[ZMIndex] - MomentInvariantRawB[ZMIndex]) * ZMWeight)

    # Calculating GeoScore    
    GeoScore = cp.sum( cp.asarray(P['GeoWeight']) * (2 * cp.abs(GeoDescriptorA - GeoDescriptorB) / (1 + cp.abs(GeoDescriptorA) + cp.abs(GeoDescriptorB))))
    
    # Calculating paper loss
    Paper_Loss = ZMScore + GeoScore
    
    # Scaled scores, fitted to shape service
    GeoScoreScaled = (6.6 - GeoScore) / 6.6 * 100.0
    ZMScoreScaled = (9.0 - ZMScore) / 9.0 * 100.0

    return GeoScoreScaled, ZMScoreScaled


def main():
    if len(sys.argv) != 4:
        print('Usage: ZMPY_CP_CLI_ShapeScore PDB_A PDB_B GridWidth')
        print('    This function takes two PDB structures and a grid width to generate shape analysis scores.')
        print('Error: You must provide exactly two input files and an input grid width.')
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Process two .pdb or .txt files.')
    parser.add_argument('input_file1', type=str, help='The first input file to process (must end with .pdb or .txt) with old PDB text format')
    parser.add_argument('input_file2', type=str, help='The second input file to process (must end with .pdb or .txt) with old PDB text format')
    parser.add_argument('grid_width', type=str, help='The third input file to process (must 0.25, 0.50 or 1.0)')

    args = parser.parse_args()

    # Perform validation checks directly after parsing arguments
    input_files = [args.input_file1, args.input_file2]
    for input_file in input_files:
        if not (input_file.endswith('.pdb') or input_file.endswith('.txt')):
            parser.error("File must end with .pdb or .txt")
        
        if not os.path.isfile(input_file):
            parser.error("File does not exist")

    try:
        GridWidth = float(args.grid_width)
    except ValueError:
        print("GridWidth cannot be converted to a float.")   

    if GridWidth not in [0.25, 0.50, 1.0]:
        parser.error("grid width must be either 0.25, 0.50, or 1.0")


    GeoScoreScaled, ZMScoreScaled=ZMPY_CP_CLI_ShapeScore(args.input_file1,args.input_file2,GridWidth)


    print('The scaled score for the geometric descriptor is calculated.')
    print(f'GeoScore {GeoScoreScaled:.2f}')

    print('The scaled score for the Zernike moments is calculated.')
    print(f'TotalZMScore {ZMScoreScaled:.2f}')

if __name__ == '__main__':
    main()