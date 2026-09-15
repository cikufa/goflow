"""Explicit paper-level RGB-D pose inference adapter (implementation class C).

Run only through scripts/bayes3d_python.sh. It uses actual Bayes3D rendering;
SMC settings and uniform-color rendering are documented implementation assumptions.
Input is an image, known mesh/calibration and a pose prior, never a true latent pose.
"""
from dataclasses import dataclass, asdict
import numpy as np
import torch  # matching CUDA runtime before JAX
import jax.numpy as jnp
import bayes3d as b


@dataclass(frozen=True)
class InferenceConfig:
    particles: int = 256
    render_batch: int = 64
    temperatures: tuple = (.125, .25, .5, 1.)
    proposal_fractions: tuple = (.15, .08, .04, .02)
    metropolis_steps: int = 2
    posterior_tempering: float = .2
    rgb_scale: float = .15
    depth_scale: float = .005
    outlier_probability: float = .01
    grid_points: tuple = (25, 25, 25)  # yaw, x, y
    grid_translation_half_fraction: float = .5


class PoseInference:
    def __init__(self, mesh, intrinsics, camera_T_world, base_rotation, support_z,
                 color_rgb, low, high, config=InferenceConfig()):
        self.cfg = config
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        if self.low.shape != (3,) or np.any(self.low >= self.high):
            raise ValueError('Prior bounds must describe [yaw, x, y]')
        self.camera_T_world = np.asarray(camera_T_world, dtype=np.float32)
        self.base_rotation = np.asarray(base_rotation, dtype=np.float32)
        self.support_z = float(support_z)
        self.color = np.asarray(color_rgb, dtype=np.float32)
        if not 0 < config.outlier_probability < 1:
            raise ValueError('Invalid pixel outlier probability')
        if min(config.rgb_scale, config.depth_scale, config.posterior_tempering) <= 0:
            raise ValueError('Likelihood scales and tempering must be positive')
        if b.RENDERER is not None:
            b.RENDERER.clear_gpu_meshmem()
        b.setup_renderer(intrinsics, num_layers=config.render_batch)
        b.RENDERER.add_mesh(mesh, mesh_name='known_object', center_mesh=False)
        self.renderer = b.RENDERER

    def render(self, poses):
        """Render hypotheses [yaw,x,y] in the known world/table coordinate frame."""
        poses = np.atleast_2d(poses)
        transform = np.repeat(np.eye(4, dtype=np.float32)[None], len(poses), axis=0)
        c, s = np.cos(poses[:, 0]), np.sin(poses[:, 0])
        rotation = np.zeros((len(poses), 3, 3), dtype=np.float32)
        rotation[:, 0, 0] = rotation[:, 1, 1] = c
        rotation[:, 0, 1], rotation[:, 1, 0], rotation[:, 2, 2] = -s, s, 1
        transform[:, :3, :3] = rotation @ self.base_rotation
        transform[:, :3, 3] = np.column_stack([poses[:, 1:3], np.full(len(poses), self.support_z)])
        transform = self.camera_T_world @ transform
        frames = []
        for start in range(0, len(poses), self.cfg.render_batch):
            chunk = transform[start:start+self.cfg.render_batch, None]
            frames.append(np.asarray(self.renderer.render_many(jnp.asarray(chunk), jnp.array([0], dtype=jnp.int32))))
        return np.concatenate(frames)

    def log_likelihood(self, poses, observed_rgb, observed_depth):
        rgb = np.asarray(observed_rgb, dtype=np.float32)
        depth = np.asarray(observed_depth, dtype=np.float32)
        if rgb.shape[:2] != depth.shape or rgb.shape[-1] != 3 or rgb.max() > 1:
            raise ValueError('Expected aligned RGB in [0,1] and depth in meters')
        observed_valid = np.isfinite(depth) & (depth > 0)
        results = []
        cfg = self.cfg
        for start in range(0, len(poses), cfg.render_batch):
            rendered = self.render(poses[start:start+cfg.render_batch])
            predicted_depth = rendered[..., 2]
            mask = np.isfinite(predicted_depth) & (predicted_depth > 0) & (rendered[..., 3] > 0)
            # Bayes3D valid rendered object pixels define C in the appendix.
            color_error = np.abs(rgb-self.color).sum(axis=-1) / cfg.rgb_scale
            depth_error = np.abs(np.where(observed_valid, depth, 0)-predicted_depth) / cfg.depth_scale
            inlier = -color_error[None]-depth_error
            inlier[:, ~observed_valid] = -np.inf
            mixture = np.logaddexp(np.log(cfg.outlier_probability),
                                   np.log1p(-cfg.outlier_probability)+inlier)
            score = np.where(mask, mixture, 0).sum(axis=(1, 2))
            score[mask.sum(axis=(1, 2)) == 0] = -np.inf
            results.append(score)
        return np.concatenate(results)

    @staticmethod
    def _weights(log_weights):
        log_weights = np.asarray(log_weights, dtype=np.float64)
        maximum = np.max(log_weights)
        if not np.isfinite(maximum):
            raise ValueError('No visible finite-likelihood pose in prior')
        weights = np.exp(log_weights-maximum)
        return weights/weights.sum()

    def infer(self, observed_rgb, observed_depth, seed):
        rng = np.random.default_rng(seed)
        cfg = self.cfg
        particles = rng.uniform(self.low, self.high, size=(cfg.particles, 3))
        likelihood = self.log_likelihood(particles, observed_rgb, observed_depth)
        weights = np.full(cfg.particles, 1/cfg.particles)
        old_temperature = 0.
        history = []
        for temperature, scale in zip(cfg.temperatures, cfg.proposal_fractions):
            temperature *= cfg.posterior_tempering
            weights = self._weights(np.log(weights)+(temperature-old_temperature)*likelihood)
            ess = float(1/np.square(weights).sum())
            # Systematic resampling and symmetric reflected random-walk MH.
            positions = (rng.random()+np.arange(cfg.particles))/cfg.particles
            indices = np.searchsorted(np.cumsum(weights), positions).clip(max=cfg.particles-1)
            particles, likelihood = particles[indices].copy(), likelihood[indices].copy()
            accepted = 0
            for _ in range(cfg.metropolis_steps):
                proposal = particles+rng.normal(size=particles.shape)*scale*(self.high-self.low)
                span = self.high-self.low
                folded = (proposal-self.low) % (2*span)
                proposal = self.low+np.minimum(folded, 2*span-folded)
                proposed_likelihood = self.log_likelihood(proposal, observed_rgb, observed_depth)
                accept = np.log(rng.random(cfg.particles)) < temperature*(proposed_likelihood-likelihood)
                particles[accept], likelihood[accept] = proposal[accept], proposed_likelihood[accept]
                accepted += int(accept.sum())
            history.append({'temperature':temperature,'ess_before_resample':ess,
                            'accepted_proposals':accepted,'map_log_likelihood':float(likelihood.max())})
            weights.fill(1/cfg.particles)
            old_temperature = temperature
        map_pose = particles[np.argmax(likelihood)].copy()
        # Retain the entire yaw range so a local grid does not invent yaw certainty.
        half = (self.high-self.low)*cfg.grid_translation_half_fraction
        grid_low, grid_high = np.maximum(self.low,map_pose-half), np.minimum(self.high,map_pose+half)
        grid_low[0], grid_high[0] = self.low[0], self.high[0]
        axes = [np.linspace(lo,hi,n,endpoint=(i != 0))
                for i,(lo,hi,n) in enumerate(zip(grid_low,grid_high,cfg.grid_points))]
        grid = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        grid_likelihood = self.log_likelihood(grid,observed_rgb,observed_depth)
        posterior = self._weights(cfg.posterior_tempering*grid_likelihood)
        mean_xy = posterior @ grid[:,1:]
        centered = grid[:,1:]-mean_xy
        covariance_xy = (centered*posterior[:,None]).T @ centered
        yaw_resultant = abs(posterior @ np.exp(1j*grid[:,0]))
        return {'particles':grid,'weights':posterior,'log_likelihood':grid_likelihood,
                'metadata':{'seed':seed,'configuration':asdict(cfg),'smc':history,
                            'map':map_pose.tolist(),'mean_xy':mean_xy.tolist(),
                            'covariance_xy':covariance_xy.tolist(),'yaw_resultant':float(yaw_resultant),
                            'grid_low':grid_low.tolist(),'grid_high':grid_high.tolist(),
                            'scope':'Bayes3D renderer with documented paper-level inference assumptions'}}
