# GPL3.0 License, JSLai
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


import numpy as np
import cupy as cp

def get_ca_distance_info_cp(xyz):

    xyz_center = cp.mean(xyz, axis=0)
    xyz_dist2center = cp.sqrt(cp.sum((xyz - xyz_center) ** 2, axis=1))

    percentiles_for_geom = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    percentile_list = cp.percentile(xyz_dist2center, percentiles_for_geom)
    percentile_list = percentile_list.reshape(-1, 1)

    mean_distance = cp.mean(xyz_dist2center)
    std_xyz_dist2center = cp.std(xyz_dist2center, ddof=1)
    n = cp.array(len(xyz_dist2center))


    skewness = (n / ((n - 1) * (n - 2))) * cp.sum(((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 3)


    fourth_moment = cp.sum(((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 4)
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * fourth_moment -
                3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    return percentile_list, std_xyz_dist2center, skewness, kurtosis
