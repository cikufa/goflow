"""Synthetic-image inference/sensitivity test, not an Isaac or planning experiment."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
import bayes3d as b
from experiments.common.bayes3d_pose import InferenceConfig, PoseInference

output=ROOT/'.cache/bayes3d-pose-probe'
output.mkdir(exist_ok=True)
truth=np.array([.2,.012,-.009])  # test generator and error calculation ONLY
color=np.array([.1,.8,.1],dtype=np.float32)
mesh=trimesh.creation.box(extents=(.04,.04,.04))
rows=[]
for noise_scale in (.0025,.005,.01):
    for label,distance in [('far',.8),('near',.25)]:
        camera=np.eye(4);camera[2,3]=distance
        cfg=InferenceConfig(particles=128,grid_points=(13,25,25),depth_scale=noise_scale)
        engine=PoseInference(mesh,b.Intrinsics(64,64,80.,80.,32.,32.,.01,2.),camera,
                             np.eye(3),0.,color,[-np.pi,-.04,-.04],[np.pi,.04,.04],cfg)
        # Generate an image, then infer using only that image and the known prior.
        image=engine.render(truth[None])[0]
        rgb=np.zeros((64,64,3),dtype=np.float32)
        rgb[image[...,3]>0]=color
        posterior=engine.infer(rgb,image[...,2],seed=321)
        metadata=posterior['metadata']
        row={'view':label,'depth_scale':noise_scale,
             'xy_error_m':float(np.linalg.norm(np.array(metadata['mean_xy'])-truth[1:])),
             'xy_variance_trace':float(np.trace(metadata['covariance_xy'])),
             'metadata':metadata}
        assert np.isfinite(posterior['weights']).all()
        assert abs(posterior['weights'].sum()-1)<1e-8
        assert row['xy_error_m']<.015,row
        key=f'{label}_{noise_scale}'
        np.savez_compressed(output/f'{key}.npz',**{k:v for k,v in posterior.items() if k!='metadata'},
                            observed_rgb=rgb,observed_depth=image[...,2])
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='metadata'}),flush=True)
for scale in (.0025,.005,.01):
    far,near=[r for r in rows if r['depth_scale']==scale]
    assert near['xy_variance_trace'] < far['xy_variance_trace'],(far,near)
summary={'status':'passed','scope':'synthetic uniformly colored box images; not original robot perception',
         'generator_truth_evaluation_only':truth.tolist(),'rows':rows}
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: finite posterior, pose recovery and closer-view uncertainty reduction across three depth scales')
