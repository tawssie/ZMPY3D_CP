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


def get_3dzd_121_descriptor02_cp(zmoment_scaled):
    zmoment_scaled[cp.isnan(zmoment_scaled)] = 0


    zmoment_scaled_norm = cp.abs(zmoment_scaled) ** 2


    zmoment_scaled_norm_positive = cp.sum(zmoment_scaled_norm, axis=2)


    zmoment_scaled_norm[:, :, 0] = 0
    zmoment_scaled_norm_negative = cp.sum(zmoment_scaled_norm, axis=2)


    zm_3dzd_invariant = cp.sqrt(zmoment_scaled_norm_positive + zmoment_scaled_norm_negative)
    zm_3dzd_invariant[zm_3dzd_invariant < 1e-20] = cp.nan

    return zm_3dzd_invariant
