"""Actual GPU renderer check in the separate Bayes3D environment."""
import json
from pathlib import Path
import time
import numpy as np
import torch  # load the matching CUDA runtime before JAX/Bayes3D
import jax
import jax.numpy as jnp
import trimesh
import bayes3d as b

start=time.time()
assert jax.devices()[0].platform == 'gpu', jax.devices()
intrinsics=b.Intrinsics(64,64,80.,80.,32.,32.,.01,2.)
b.setup_renderer(intrinsics,num_layers=8)
b.RENDERER.add_mesh(trimesh.creation.box(extents=(.04,.04,.04)),mesh_name='probe_box')
poses=jnp.eye(4)[None,None,:,:].repeat(2,axis=0)
poses=poses.at[:,0,2,3].set(.5)
poses=poses.at[1,0,0,3].set(.04)
render=np.asarray(b.RENDERER.render_many(poses,jnp.array([0],dtype=jnp.int32)))
assert render.shape == (2,64,64,4), render.shape
mask=(render[...,2] > 0) & (render[...,2] < 1)
assert np.all(mask.sum(axis=(1,2)) > 10), mask.sum(axis=(1,2))
assert np.isfinite(render[mask]).all()
centers=[float(np.nonzero(m)[1].mean()) for m in mask]
assert centers[1] > centers[0] + 3, centers
output=Path('.cache/bayes3d-renderer-probe')
output.mkdir(exist_ok=True)
np.savez_compressed(output/'render.npz',render=render)
result={'status':'passed','bayes3d_commit':'4e2919dd82c4596b7baca570a15bb7f3a89566a4',
        'torch':torch.__version__,'jax':jax.__version__,'device':str(jax.devices()[0]),
        'shape':list(render.shape),'foreground_pixels':mask.sum(axis=(1,2)).tolist(),
        'horizontal_centers_px':centers,'wall_seconds':time.time()-start,
        'scope':'Bayes3D GPU renderer only; not a validated RGB-D perception pipeline'}
(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
