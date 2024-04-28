# GPL3.0 License, ZM
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

def get_bbox_moment_xyz_sample_01_cp(center, radius, dimension_bbox_scaled):

    x_edge, y_edge, z_edge = dimension_bbox_scaled
    

    x_sample = (cp.arange(x_edge + 1) - center[0]) / radius
    y_sample = (cp.arange(y_edge + 1) - center[1]) / radius
    z_sample = (cp.arange(z_edge + 1) - center[2]) / radius


    return x_sample, y_sample, z_sample

