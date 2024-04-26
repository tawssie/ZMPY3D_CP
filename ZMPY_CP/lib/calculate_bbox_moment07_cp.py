# GPL3.0 License, JSL from ZM, this is my originality, to calculate 3D bbox moment by tensordot
#
# Copyright (C) 2024  Jhih-Siang Lai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import cupy as cp

def calculate_bbox_moment07_cp(Voxel3D, MaxOrder, X_sample, Y_sample, Z_sample):

    p, q, r = cp.meshgrid(
        cp.arange(1, MaxOrder + 2, dtype=cp.float64),
        cp.arange(1, MaxOrder + 2, dtype=cp.float64),
        cp.arange(1, MaxOrder + 2, dtype=cp.float64),
        indexing='ij'
    )

    extended_voxel_3d = cp.pad(Voxel3D, ((0, 1), (0, 1), (0, 1)))

    diff_extended_voxel_3d = cp.diff(cp.diff(cp.diff(extended_voxel_3d, axis=0), axis=1), axis=2)

    x_power = cp.power(X_sample[1:, cp.newaxis], cp.arange(1, MaxOrder + 2, dtype=cp.float64))
    y_power = cp.power(Y_sample[1:, cp.newaxis], cp.arange(1, MaxOrder + 2, dtype=cp.float64))
    z_power = cp.power(Z_sample[1:, cp.newaxis], cp.arange(1, MaxOrder + 2, dtype=cp.float64))

    bbox_moment = cp.tensordot(
        z_power,
        cp.tensordot(
            y_power,
            cp.tensordot(x_power, diff_extended_voxel_3d, axes=([0], [0])),
            axes=([0], [1])
        ),
        axes=([0], [2])
    )

    bbox_moment = -cp.transpose(bbox_moment, (2, 1, 0)) / (p * q * r)
    volume_mass = bbox_moment[0, 0, 0]
    center = bbox_moment[[1, 0, 0], [0, 1, 0], [0, 0, 1]] / volume_mass

    return volume_mass, center, bbox_moment

