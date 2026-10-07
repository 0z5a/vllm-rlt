"""Compile the six queued BF16-output projection shapes without GPU execution."""
import base64
import hashlib
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tmem_gemm import project

output = Path(__file__).parent / 'compile'
output.mkdir()
records = []
for m, n, k in ((1,2048,2048),(32,2048,2048),(64,2048,2048),
                (512,2048,2048),(32,5632,2048),(32,2048,5632)):
    source = ASTSource(project, {'A':'*bf16','W':'*bf16','C':'*bf16'},
                       constexprs={'M':m,'N':n,'K':k,'BM':64,'BN':64,'BK':64})
    kernel = triton.compile(source,target=GPUTarget('cuda',110,32),
                            options={'num_warps':4,'num_stages':2})
    ptx = kernel.asm['ptx']
    instructions = {name:ptx.count(name) for name in
                    ('tcgen05.alloc','tcgen05.mma','tcgen05.ld','tcgen05.dealloc')}
    assert all(instructions.values()) and '.target sm_110a' in ptx
    name = f'm{m}-n{n}-k{k}.ptx'
    (output/name).write_text(ptx)
    records.append({'shape':[m,n,k],'input':'BF16','output':'BF16',
                    'accumulator':'FP32','instructions':instructions,
                    'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
                    'name':name,'ptx_base64':base64.b64encode(ptx.encode()).decode()})
result = {'scope':'OFFLINE_COMPILE_ONLY_NOT_NUMERICAL_OR_SPEED_TEST',
          'triton':triton.__version__,'gpu_executed':False,'records':records}
(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
