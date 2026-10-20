"""Compiler-side Hopper tensorization intrinsics missing from the TVM pin."""


def register_wgmma(tvm):
    from tvm.script import tirx as T

    @T.prim_func(s_tir=True)
    def desc(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (64, 16), 'float16', scope='shared', offset_factor=1)
        B = T.match_buffer(b, (16, 64), 'float16', scope='shared', offset_factor=1)
        C = T.match_buffer(c, (64, 64), 'float32', scope='local', offset_factor=1)
        with T.sblock('root'):
            T.reads(C[0:64, 0:64], A[0:64, 0:16], B[0:16, 0:64])
            T.writes(C[0:64, 0:64])
            for i, j, k in T.grid(64, 64, 16):
                with T.sblock('update'):
                    vi, vj, vk = T.axis.remap('SSR', [i, j, k])
                    C[vi, vj] = C[vi, vj] + T.Cast('float32', A[vi, vk]) * T.Cast('float32', B[vk, vj])

    @T.prim_func(s_tir=True)
    def impl(a: T.handle, b: T.handle, c: T.handle):
        a0, a1, b0, b1, c0, c1 = T.int32(), T.int32(), T.int32(), T.int32(), T.int32(), T.int32()
        A = T.match_buffer(a, (64, 16), 'float16', scope='shared', strides=[a0,a1], offset_factor=1)
        B = T.match_buffer(b, (16, 64), 'float16', scope='shared', strides=[b0,b1], offset_factor=1)
        C = T.match_buffer(c, (64, 64), 'float32', scope='local', strides=[c0,c1], offset_factor=1)
        with T.sblock('root'):
            T.reads(C[0:64, 0:64], A[0:64, 0:16], B[0:16, 0:64])
            T.writes(C[0:64, 0:64])
            # A compiler intrinsic boundary. LowerHopper resolves descriptors,
            # register distribution, fence/commit/wait and native PTX calls.
            T.evaluate(T.call_extern('int32', 'gtsim.hopper.m64n64k16',
                                    T.address_of(A[0,0]), T.address_of(B[0,0]), T.address_of(C[0,0])))

    tvm.s_tir.TensorIntrin.register('gtsim_hopper_m64n64k16', desc, impl, override=True)
    return desc, impl
