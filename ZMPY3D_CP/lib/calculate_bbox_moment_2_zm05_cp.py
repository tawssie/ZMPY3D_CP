# GPL3.0, JSL from ZM
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
import numpy as np

def calculate_bbox_moment_2_zm05_cp(max_order, gcache_complex, gcache_pqr_linear, gcache_complex_index, clm_cache_3d, bbox_moment):

    max_n = max_order.get() + 1

    bbox_moment = np.reshape(cp.transpose(bbox_moment, (2, 1, 0)), -1)

    zm_geo = gcache_complex * bbox_moment[gcache_pqr_linear - 1]

    zm_geo_sum = cp.zeros(max_n * max_n * max_n, dtype=cp.complex128)

    cp.add.at(zm_geo_sum.real, gcache_complex_index - 1, zm_geo.real)
    cp.add.at(zm_geo_sum.imag, gcache_complex_index - 1, zm_geo.imag)

    zm_geo_sum[zm_geo_sum == 0.0] = cp.nan

    zmoment_raw = zm_geo_sum * (3.0 / (4.0 * cp.pi))
    zmoment_raw = zmoment_raw.reshape((max_n, max_n, max_n))
    zmoment_raw = cp.transpose(zmoment_raw, (2, 1, 0))
    zmoment_scaled = zmoment_raw * clm_cache_3d

    return zmoment_scaled, zmoment_raw

