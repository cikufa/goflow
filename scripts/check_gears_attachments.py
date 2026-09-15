"""Inspect actual USD grasp constraints across cloned Gears environments."""
import json
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common.runtime import kit_arguments
kit = kit_arguments()
from omni.isaac.lab.app import AppLauncher
app = AppLauncher(headless=True, kit_args=kit).app
try:
    from goflow.environments.med_gear.direct_panda_position import MyPandaEnv, MyPandaEnvCfg
    from pxr import UsdPhysics
    import omni.usd
    cfg = MyPandaEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 0
    env = MyPandaEnv(cfg)
    env.reset()
    rows = []
    for prim in omni.usd.get_context().get_stage().Traverse():
        if prim.IsA(UsdPhysics.FixedJoint) and 'AssemblerFixedJoint' in str(prim.GetPath()):
            joint = UsdPhysics.FixedJoint(prim)
            rows.append({'path': str(prim.GetPath()),
                         'body0': [str(x) for x in joint.GetBody0Rel().GetTargets()],
                         'body1': [str(x) for x in joint.GetBody1Rel().GetTargets()],
                         'local_pos0': list(joint.GetLocalPos0Attr().Get()),
                         'local_pos1': list(joint.GetLocalPos1Attr().Get())})
    output = ROOT / 'results/original_gears' / os.environ.get('GOFLOW_ATTACHMENT_REPORT', 'attachments_before_fix.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2), flush=True)
    env.close()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)
app.close()
