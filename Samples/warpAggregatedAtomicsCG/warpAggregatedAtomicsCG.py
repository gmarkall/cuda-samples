# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of NVIDIA CORPORATION nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from numba import cuda, njit
import numpy as np


NUM_ELEMS = 10000000


@cuda.jit(device=True)
def atomicAggInc(counter):
    active = cuda.cg.coalesced_threads()

    mask = active.ballot(True)
    # Select the leader
    leader = cuda.ffs(mask) - 1

    # Leader does the update
    if active.thread_rank == leader:
        res = cuda.atomic.add(counter, 0, cuda.popc(mask))

    # Broadcast result
    res = active.shfl(res, leader)

    # Each thread computes its own value
    return res + cuda.popc(mask & ((1 << active.thread_rank) - 1))


@cuda.jit
def filter_arr(dst, nres, src, n):
    tid = cuda.grid(1)
    step = cuda.gridsize(1)

    for i in range(tid, n, step):
        if src[i] > 0:
            dst[atomicAggInc(nres)] = src[i]


@njit
def host_filter_arr(dst, src):
    counter = 0

    for i in range(len(src)):
        if src[i] > 0:
            dst[counter] = src[i]
            counter += 1

    return counter


def main():
    dev = cuda.devices.get_context().device
    print('Device %s: %s with CC %s.%s' % (dev.id, dev.name.decode('utf-8'),
                                           *dev.compute_capability))

    data_to_filter = np.random.randint(20, size=NUM_ELEMS)
    dev_filtered_data = np.zeros_like(data_to_filter)
    host_filtered_data = np.zeros_like(data_to_filter)
    nres = np.zeros(1)

    nthreads = 512
    nblocks = NUM_ELEMS // nthreads
    filter_arr[nblocks, nthreads](dev_filtered_data, nres, data_to_filter,
                                  NUM_ELEMS)

    host_nres = host_filter_arr(host_filtered_data, data_to_filter)

    if nres[0] != host_nres:
        print("FAILED: Host count %d != device count %d" % (host_nres,
                                                            nres[0]))
        return 1

    if np.any(dev_filtered_data != host_filtered_data):
        print("FAILED: Host and device data differ")
        return 1

    print("PASSED")


if __name__ == '__main__':
    main()
