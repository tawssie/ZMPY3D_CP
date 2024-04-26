# GPL3.0 License, JSL, ZM
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

def calculate_zm_by_ab_rotation_01_cp(
        zmoment_raw, binomial_cache, ab_list, max_order, clm_cache,
        s_id, n, l, m, mu, k, is_nlm_value):

    a = ab_list[:, 0]
    b = ab_list[:, 1]


    aac = cp.real(a * cp.conj(a)).astype(cp.complex128)
    bbc = cp.real(b * cp.conj(b)).astype(cp.complex128)
    bbcaac = -bbc / aac


    abc = -(a / cp.conj(b))
    ab = a / b


    bbcaac_pow_k_list = cp.log(bbcaac)[:, None] * cp.arange(max_order + 1)
    aac_pow_l_list = cp.log(aac)[:, None] * cp.arange(max_order + 1)
    ab_pow_m_list = cp.log(ab)[:, None] * cp.arange(max_order + 1)
    abc_pow_mu_list = cp.log(abc)[:, None] * cp.arange(-max_order.get(), max_order + 1)

    F_exp = cp.zeros(len(s_id), dtype=cp.complex128)

    F_exp[mu >= 0] = zmoment_raw[n[mu >= 0], l[mu >= 0], mu[mu >= 0]]
    F_exp[(mu < 0) & (mu % 2 == 0)] = cp.conj(zmoment_raw[n[(mu < 0) & (mu % 2 == 0)], l[(mu < 0) & (mu % 2 == 0)], -mu[(mu < 0) & (mu % 2 == 0)]])
    F_exp[(mu < 0) & (mu % 2 != 0)] = -cp.conj(zmoment_raw[n[(mu < 0) & (mu % 2 != 0)], l[(mu < 0) & (mu % 2 != 0)], -mu[(mu < 0) & (mu % 2 != 0)]])


    F_exp = cp.log(F_exp)

    max_n = max_order + 1
    clm = clm_cache[l * max_n + m].astype(cp.complex128).flatten()
    bin = (binomial_cache[l - mu, k - mu] + binomial_cache[l + mu, k - m]).astype(cp.complex128).flatten()


    al = aac_pow_l_list[:, l]
    abpm = ab_pow_m_list[:, m]
    amu = abc_pow_mu_list[:, max_order + mu]
    bbk = bbcaac_pow_k_list[:, k]

    nlm = F_exp + al + clm + abpm + amu + bbk + bin

    z_nlm = cp.zeros(is_nlm_value.shape, dtype=cp.complex128).reshape(-1, 1)
    z_nlm = cp.tile(z_nlm, (1, a.size))

    exp_nlm = cp.exp(nlm).transpose()


    cp.add.at(z_nlm.real, s_id, exp_nlm.real)
    cp.add.at(z_nlm.imag, s_id, exp_nlm.imag)

    zm = cp.full((np.prod(zmoment_raw.shape), a.size), cp.nan, dtype=cp.complex128)
    zm[is_nlm_value] = z_nlm

    zm = cp.reshape(zm, zmoment_raw.shape + (a.size,))
    zm = cp.transpose(zm, (2, 1, 0, 3))

    return zm


