# Bayes3D feasibility audit

Inspected official https://github.com/probcomp/bayes3d at
`4e2919dd82c4596b7baca570a15bb7f3a89566a4`, cloned into ignored `.deps/bayes3d`.

The README recommends Python 3.9, Torch 2.2.0/cu118 and JAX 0.4.20 with local CUDA 11 support. Its metadata supports Python >=3.9, but pins `genjax==0.1.1`. On 2026-09-15, a real dependency retrieval attempt failed:

```bash
scripts/project_python.sh -m pip download --no-deps genjax==0.1.1 \
  --dest .cache/bayes3d-wheels --index-url https://pypi.org/simple
```

PyPI has no 0.1.1 artifacts; currently listed stable GenJAX versions require Python >=3.11. Raw output is `.cache/bayes3d-dependency-probe.log`. The Python 3.10 simulator environment was not changed. EGL/GLU headers, GCC and Ninja already exist, so missing system graphics headers are not the demonstrated blocker. The host CUDA compiler is 11.5 whereas Torch uses runtime 11.8; compiled-extension compatibility has not yet been tested.

This demonstrates that the declared installation cannot currently resolve unchanged. It does **not** prove the renderer/likelihood subset cannot run with a source dependency or separate process. GenJAX is imported by Bayes3D's generative-model modules, not directly by its renderer. Such a subset would require an explicit integration audit. No Bayes3D runtime or fallback perception success is claimed at this stage.

No private credentials, sudo, system-driver change or asset-pack download was attempted.
