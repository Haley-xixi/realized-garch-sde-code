import sys
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "rv.pkl"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


# =========================================================
# Local utilities
# =========================================================
try:
    from SDE_Garch_utils.utils1 import build_data_dict, set_seed_all, load_rv
except ImportError as exc:
    raise ImportError(
        "Could not import SDE_Garch_utils.utils1. "
        "Check that SDE_Garch_utils/utils1.py exists and is importable."
    ) from exc


# =========================================================
# Global setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed_all(seed=1313)
sys.setrecursionlimit(2500)


LossType = Literal["joint_qnll", "composite_nll", "return_qnll","return_rv"]
TransformType = Literal["exp", "softplus"]


# =========================================================
# Data helper
# =========================================================
def get_ou_arrays(
    data_dict: Dict,
    ticker: str,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract finite train/test log-price and realised-variance arrays."""
    data = data_dict[ticker]["train"]

    y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
    rv_train = np.asarray(data["rv_train"], dtype=np.float64).reshape(-1)
    rv_train = np.maximum(rv_train, eps)

    train_mask = np.isfinite(y_train) & np.isfinite(rv_train)
    y_train = y_train[train_mask]
    rv_train = rv_train[train_mask]

    y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
    rv_test = np.asarray(data["rv_test"], dtype=np.float64).reshape(-1)
    rv_test = np.maximum(rv_test, eps)

    test_mask = np.isfinite(y_test) & np.isfinite(rv_test)
    y_test = y_test[test_mask]
    rv_test = rv_test[test_mask]

    return y_train, rv_train, y_test, rv_test


def positive(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(raw) + eps


def inverse_softplus(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable inverse of softplus on positive values."""
    value = value.clamp_min(eps)
    return value + torch.log(-torch.expm1(-value))


def safe_std(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x.std(unbiased=False).clamp_min(eps)


# =========================================================
# Positive continuous-time state system
# =========================================================
class PositiveJointSDE(torchsde.SDEIto):
    r"""
    State vector
    ------------
        X_t = (Y_t, x_t, z_t)

    Positive economic states
    ------------------------
        s_t = g_s(x_t) > 0
        R_t = g_R(z_t) > 0

    Full dynamics
    -------------
        dY_t
            = sqrt(s_t) dW_t^Y

        dx_t
            = [
                  kappa * (theta - x_t)
                  + gamma * z_t
              ] dt
              + eta_s dW_t^s

        dz_t
            = kappa_z * (
                  xi
                  + phi * x_t
                  - z_t
              ) dt
              + eta_R dW_t^R
              + sigma_u dU_t

    Feedback specification
    ----------------------
    The model uses NAIVE realized-measure feedback:

        gamma * z_t = gamma * log(R_t).

    No centering term such as (z_t - xi - phi*theta) is used.

    Full-model correlations
    -----------------------
        Corr(dW^Y, dW^s) = rho_Ys
        Corr(dW^Y, dW^R) = rho_YR
        Corr(dW^s, dW^R) = 0

        U is independent.

    No-realized-feedback ablation
    -----------------------------
    When

        gamma_fixed_zero = True

    impose

        gamma = 0,

    so the latent log-variance drift becomes

        dx_t = kappa * (theta - x_t) dt
               + eta_s dW_t^s.

    The realized-measure state and Brownian dependence structure
    otherwise remain unchanged.

    Independent-Brownian ablation
    -----------------------------
    When

        independent_brownian = True

    impose

        rho_Ys = 0
        rho_YR = 0

    while retaining

        eta_s > 0
        eta_R > 0
        sigma_u > 0.

    Therefore

        dY_t = 0.01 sqrt(s_t) dW_t^Y

        dx_t = [...]
               + eta_s dW_t^s

        dz_t = kappa_z*(xi + phi*x_t - z_t)dt
               + eta_R dW_t^R
               + sigma_u dU_t

    with W^Y, W^s, W^R, and U mutually independent.

    """

    noise_type = "general"
    sde_type = "ito"

    def __init__(
        self,
        transform: TransformType = "exp",
        gamma_fixed_zero: bool = False,
        independent_brownian: bool = False,
    ):
        super().__init__(noise_type=self.noise_type)

        self.gamma_fixed_zero = gamma_fixed_zero
        self.independent_brownian = independent_brownian

        if (
            self.gamma_fixed_zero
            and self.independent_brownian
        ):
            raise ValueError(
                "GAMMA_FIXED_ZERO and INDEPENDENT_BROWNIAN "
                "are separate ablations and should not both be True."
            )

        if transform not in (
            "exp",
            "softplus",
        ):
            raise ValueError(
                "transform must be 'exp' or 'softplus'."
            )

        self.transform = transform

        # =====================================================
        # x-state parameters
        # =====================================================

        self._kappa = nn.Parameter(
            torch.tensor(
                0.20,
                dtype=torch.float32,
            )
        )

        self._theta = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self._gamma = nn.Parameter(
            torch.tensor(
                0.05,
                dtype=torch.float32,
            )
        )

        self._eta_s = nn.Parameter(
            torch.tensor(
                0.10,
                dtype=torch.float32,
            )
        )

        # =====================================================
        # z-state parameters
        # =====================================================

        self._kappa_z = nn.Parameter(
            torch.tensor(
                0.20,
                dtype=torch.float32,
            )
        )

        self._xi = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self._phi = nn.Parameter(
            torch.tensor(
                0.05,
                dtype=torch.float32,
            )
        )

        self._eta_R = nn.Parameter(
            torch.tensor(
                0.10,
                dtype=torch.float32,
            )
        )

        self._sigma_u = nn.Parameter(
            torch.tensor(
                0.10,
                dtype=torch.float32,
            )
        )


        self._rho_Ys_raw = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self._rho_YR_raw = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )


        # =====================================================
        # No-realized-feedback ablation
        #
        # gamma = 0
        # =====================================================

        if self.gamma_fixed_zero:

            with torch.no_grad():
                self._gamma.zero_()

            self._gamma.requires_grad_(
                False
            )

        # =====================================================
        # Independent-Brownian ablation
        #
        # rho_Ys = rho_YR = 0
        # =====================================================

        if self.independent_brownian:

            self._rho_Ys_raw.requires_grad_(
                False
            )

            self._rho_YR_raw.requires_grad_(
                False
            )
    # =========================================================
    # Structural parameters
    # =========================================================

    @property
    def kappa(
        self,
    ) -> torch.Tensor:

        return positive(
            self._kappa
        )
    @property
    def kappa_z(
        self,
    ) -> torch.Tensor:

        return positive(
            self._kappa_z
        )

    @property
    def kappa_s(
        self,
    ) -> torch.Tensor:
        return self.kappa

    @property
    def theta(
        self,
    ) -> torch.Tensor:

        return self._theta

    @property
    def gamma(
        self,
    ) -> torch.Tensor:

        if self.gamma_fixed_zero:
            return torch.zeros_like(
                self._gamma
            )

        return self._gamma

    # =========================================================
    # Diffusion parameters
    # =========================================================

    @property
    def eta_s(
        self,
    ) -> torch.Tensor:

        return positive(
            self._eta_s
        )

    @property
    def eta_R(
        self,
    ) -> torch.Tensor:

        return positive(
            self._eta_R
        )

    @property
    def sigma_u(
        self,
    ) -> torch.Tensor:

        return positive(
            self._sigma_u
        )

    # =========================================================
    # z-state drift parameters
    # =========================================================

    @property
    def xi(
        self,
    ) -> torch.Tensor:

        return self._xi

    @property
    def phi(
        self,
    ) -> torch.Tensor:

        return self._phi

    # =========================================================
    # Correlations
    # =========================================================

    def correlations(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        margin = 0.995


        if self.independent_brownian:

            rho_Ys = torch.zeros_like(
                self._rho_Ys_raw
            )

            rho_YR = torch.zeros_like(
                self._rho_YR_raw
            )

            return (
                rho_Ys,
                rho_YR,
            )



        rho_Ys = (
            -margin
            * torch.sigmoid(
                self._rho_Ys_raw
            )
        )

        remaining = torch.sqrt(
            (
                1.0
                - rho_Ys.square()
            ).clamp_min(
                1e-8
            )
        )

        rho_YR = (
            -margin
            * remaining
            * torch.sigmoid(
                self._rho_YR_raw
            )
        )

        return (
            rho_Ys,
            rho_YR,
        )

    @property
    def rho_Ys(
        self,
    ) -> torch.Tensor:

        return self.correlations()[0]

    @property
    def rho_YR(
        self,
    ) -> torch.Tensor:

        return self.correlations()[1]

    # =========================================================
    # Positive state transformations
    # =========================================================

    def variance_transform(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if self.transform == "exp":

            return torch.exp(
                x.clamp(
                    min=-15.0,
                    max=15.0,
                )
            )

        return (
            F.softplus(x)
            + 1e-8
        )

    def realised_transform(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        if self.transform == "exp":

            return torch.exp(
                z.clamp(
                    min=-15.0,
                    max=15.0,
                )
            )

        return (
            F.softplus(z)
            + 1e-8
        )

    def inverse_variance_transform(
        self,
        s: torch.Tensor,
    ) -> torch.Tensor:

        s = s.clamp_min(
            1e-8
        )

        if self.transform == "exp":

            return torch.log(
                s
            )

        return inverse_softplus(
            s
        )

    def inverse_realised_transform(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:

        R = R.clamp_min(
            1e-8
        )

        if self.transform == "exp":

            return torch.log(
                R
            )

        return inverse_softplus(
            R
        )

    # =========================================================
    # Drift
    # =========================================================

    def f(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:

        del t

        Y, x, z = state.unbind(dim=-1)
        del Y

        dY = torch.zeros_like(x)

        # Naive realized-measure feedback:
        # dx = kappa(theta - x) + gamma*z
        #
        # Here z_t = log(R_t). The realized-measure state enters
        # the latent log-variance drift directly, without centering.
        naive_feedback = z

        dx = (
            self.kappa * (self.theta - x)
            + self.gamma * naive_feedback
        )

        dz = (
            self.kappa_z
            * (
                self.xi
                + self.phi * x
                - z
            )
        )

        return torch.stack(
            (dY, dx, dz),
            dim=-1,
        )

    # =========================================================
    # Diffusion
    # =========================================================

    def g(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:

        del t

        Y, x, z = state.unbind(
            dim=-1
        )

        del Y, z

        # -----------------------------------------------------
        # Positive latent variance
        # -----------------------------------------------------

        s = self.variance_transform(
            x
        )

        sqrt_s = torch.sqrt(
            s.clamp_min(
                1e-8
            )
        )

        # -----------------------------------------------------
        # Correlations
        # -----------------------------------------------------

        rho_Ys, rho_YR = (
            self.correlations()
        )

        ys_residual = torch.sqrt(
            (
                1.0
                - rho_Ys.square()
            ).clamp_min(
                1e-8
            )
        )

        # -----------------------------------------------------
        # Construct W^R so Corr(W^s, W^R) = 0
        # -----------------------------------------------------

        wr_e2 = (
            -rho_Ys
            * rho_YR
            / ys_residual
        )

        wr_e3 = torch.sqrt(
            (
                1.0
                - rho_YR.square()
                - wr_e2.square()
            ).clamp_min(
                1e-8
            )
        )

        zeros = torch.zeros_like(
            x
        )

        # -----------------------------------------------------
        # dY
        # -----------------------------------------------------

        gY = torch.stack(
            (
                0.01 * sqrt_s,
                zeros,
                zeros,
                zeros,
            ),
            dim=-1,
        )

        # -----------------------------------------------------
        # dx
        # -----------------------------------------------------

        gx = torch.stack(
            (
                self.eta_s
                * rho_Ys.expand_as(x),

                self.eta_s
                * ys_residual.expand_as(x),

                zeros,
                zeros,
            ),
            dim=-1,
        )

        # -----------------------------------------------------
        # dz
        # -----------------------------------------------------

        gz = torch.stack(
            (
                self.eta_R
                * rho_YR.expand_as(x),

                self.eta_R
                * wr_e2.expand_as(x),

                self.eta_R
                * wr_e3.expand_as(x),

                self.sigma_u.expand_as(x),
            ),
            dim=-1,
        )

        return torch.stack(
            (
                gY,
                gx,
                gz,
            ),
            dim=1,
        )

# =========================================================
# Training-standardisation statistics
# =========================================================
def compute_target_stats(
    Y_obs: np.ndarray,
    R_obs: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Statistics are computed from training data only."""
    returns = 100.0 * np.diff(np.asarray(Y_obs, dtype=np.float64))
    log_rv = np.log(np.maximum(np.asarray(R_obs, dtype=np.float64), eps))

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(max(np.std(returns), 1e-6)),
        "log_rv_mean": float(np.mean(log_rv)),
        "log_rv_std": float(max(np.std(log_rv), 1e-6)),
    }


def standardise_joint_targets(
    return_values: torch.Tensor,
    log_rv_values: torch.Tensor,
    stats: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    r = (return_values - stats["return_mean"]) / stats["return_std"]
    v = (log_rv_values - stats["log_rv_mean"]) / stats["log_rv_std"]
    return r, v


# =========================================================
# Predictive losses
# =========================================================
def rv_qlike_loss(
    predicted_integrated_variance: torch.Tensor,
    rv_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    H = predicted_integrated_variance.clamp_min(eps)
    RV = rv_true.clamp_min(eps)

    return (
        torch.log(H)
        + RV / H
    ).mean()
def return_rv_loss(
    predicted_integrated_variance: torch.Tensor,
    return_true: torch.Tensor,
    rv_true: torch.Tensor,
    lambda_rv: float = 0.001,
    eps: float = 1e-8,
    return_components: bool = False,
):
    H = predicted_integrated_variance.clamp_min(eps)

    return_loss = 0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(H)
        + return_true.square() / H
    ).mean()

    rv_loss = (
        torch.log(H)
        + rv_true.clamp_min(eps) / H
    ).mean()

    total_loss = return_loss + lambda_rv * rv_loss

    if return_components:
        return total_loss, return_loss, rv_loss

    return total_loss


def joint_gaussian_qnll(
    return_particles: torch.Tensor,
    log_rv_particles: torch.Tensor,
    return_true: torch.Tensor,
    log_rv_true: torch.Tensor,
    covariance_jitter: float = 1e-4,
) -> torch.Tensor:
    """Joint Gaussian quasi-NLL from a Monte Carlo predictive ensemble."""
    if return_particles.shape != log_rv_particles.shape:
        raise ValueError("Particle arrays must have the same shape.")

    M, _ = return_particles.shape
    if M < 2:
        raise ValueError("joint_gaussian_qnll requires at least two particles.")

    particles = torch.stack((return_particles, log_rv_particles), dim=-1)
    mean = particles.mean(dim=0)
    centered = particles - mean.unsqueeze(0)

    covariance = torch.einsum("mbi,mbj->bij", centered, centered) / (M - 1)

    eye = torch.eye(2, dtype=particles.dtype, device=particles.device).unsqueeze(0)
    covariance = covariance + covariance_jitter * eye

    observed = torch.stack((return_true, log_rv_true), dim=-1)
    error = observed - mean

    chol, info = torch.linalg.cholesky_ex(covariance)
    if torch.any(info > 0):
        covariance = covariance + 10.0 * covariance_jitter * eye
        chol = torch.linalg.cholesky(covariance)

    solved = torch.cholesky_solve(error.unsqueeze(-1), chol).squeeze(-1)
    mahalanobis = (error * solved).sum(dim=-1)

    diagonal = torch.diagonal(chol, dim1=-2, dim2=-1)
    log_det = 2.0 * torch.log(diagonal).sum(dim=-1)

    return 0.5 * (
        2.0 * math.log(2.0 * math.pi) + log_det + mahalanobis
    ).mean()


def gaussian_kde_log_prob(
    samples: torch.Tensor,
    target: torch.Tensor,
    min_bandwidth: float = 0.05,
) -> torch.Tensor:
    """Differentiable Gaussian-kernel mixture log density."""
    M = samples.shape[0]
    if M < 2:
        raise ValueError("KDE loss requires at least two particles.")

    sample_std = samples.std(dim=0, unbiased=True)
    bandwidth = (
        1.06 * sample_std * (float(M) ** (-1.0 / 5.0))
    ).clamp_min(min_bandwidth)

    z = (target.unsqueeze(0) - samples) / bandwidth.unsqueeze(0)
    component_log_prob = (
        -0.5 * z.square()
        - torch.log(bandwidth).unsqueeze(0)
        - 0.5 * math.log(2.0 * math.pi)
    )

    return torch.logsumexp(component_log_prob, dim=0) - math.log(M)


def composite_predictive_nll(
    return_particles: torch.Tensor,
    log_rv_particles: torch.Tensor,
    return_true: torch.Tensor,
    log_rv_true: torch.Tensor,
    lambda_rv: float = 0.01,
) -> torch.Tensor:
    log_p_return = gaussian_kde_log_prob(return_particles, return_true)
    log_p_log_rv = gaussian_kde_log_prob(log_rv_particles, log_rv_true)
    return -(log_p_return + lambda_rv * log_p_log_rv).mean()


def return_qnll(
    predicted_integrated_variance: torch.Tensor,
    return_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    H = predicted_integrated_variance.clamp_min(eps)
    return 0.5 * (
        math.log(2.0 * math.pi) + torch.log(H) + return_true.square() / H
    ).mean()


# =========================================================
# Simulation helpers
# =========================================================
def initialise_state(
    model: PositiveJointSDE,
    Y0: torch.Tensor,
    R0: torch.Tensor,
    init_mode: str,
) -> torch.Tensor:
    R0 = R0.clamp_min(1e-8)

    if init_mode == "from_rv":
        x0 = model.inverse_variance_transform(R0)
    elif init_mode == "long_run":
        x0 = model.theta.expand_as(R0)
    else:
        raise ValueError("init_mode must be 'from_rv' or 'long_run'.")

    # The realised-measure state starts from the observed current RV.
    z0 = model.inverse_realised_transform(R0)
    return torch.cat((Y0, x0, z0), dim=-1)


def simulate_particles(
    model: PositiveJointSDE,
    X0: torch.Tensor,
    ts: torch.Tensor,
    n_particles: int,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        path:             (T, M, B, 3)
        return_particles: (M, B)
        log_R_particles:  (M, B)
        H_particles:      (M, B)
    """
    B, D = X0.shape

    X0_particles = (
        X0.unsqueeze(0)
        .expand(n_particles, B, D)
        .reshape(n_particles * B, D)
        .contiguous()
    )

    raw_path = torchsde.sdeint(
        model,
        X0_particles,
        ts,
        method="euler",
        dt=dt,
    )

    path = raw_path.view(len(ts), n_particles, B, D)

    Y0_particles = X0[:, 0].unsqueeze(0).expand(n_particles, B)
    Y1_particles = path[-1, :, :, 0]
    return_particles = 100.0 * (Y1_particles - Y0_particles)

    z1_particles = path[-1, :, :, 2]
    R1_particles = model.realised_transform(z1_particles)
    log_R_particles = torch.log(R1_particles.clamp_min(1e-8))

    x_path = path[:, :, :, 1]
    s_path = model.variance_transform(x_path)

    # Trapezoidal approximation of H = integral s_t dt.
    time_deltas = (ts[1:] - ts[:-1]).view(-1, 1, 1)
    H_particles = (
        0.5 * (s_path[:-1] + s_path[1:]) * time_deltas
    ).sum(dim=0)

    return path, return_particles, log_R_particles, H_particles


# =========================================================
# Training
# =========================================================
def train(
    ticker: str,
    Y_obs: np.ndarray,
    R_obs: np.ndarray,
    init_mode: str = "from_rv",
    transform: TransformType = "exp",
    loss_type: LossType = "return_rv",
    forecast_horizon: int = 1,
    n_substeps: int = 10,
    n_particles: int = 64,
    n_epochs: int = 3000,
    lr: float = 1e-4,
    batch_size: int = 32,
    lambda_rv: float = 0.1,
    covariance_jitter: float = 1e-4,
    grad_clip: float = 5.0,
    device: Optional[torch.device] = None,
    resume: bool = True,
    initial_state_dict=None,
    gamma_fixed_zero: bool = False,
    independent_brownian: bool = False,

) -> Tuple[PositiveJointSDE, List[float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")
    if n_substeps < 1:
        raise ValueError("n_substeps must be at least 1.")
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2.")

    Y_obs = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    R_obs = np.maximum(np.asarray(R_obs, dtype=np.float64).reshape(-1), 1e-8)

    if len(Y_obs) != len(R_obs):
        raise ValueError("Y_obs and R_obs must have the same length.")
    if len(Y_obs) <= forecast_horizon:
        raise ValueError("The time series is too short for the requested horizon.")

    target_stats = compute_target_stats(Y_obs, R_obs)

    model = PositiveJointSDE(
        transform=transform,
        gamma_fixed_zero=gamma_fixed_zero,
        independent_brownian=independent_brownian,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if gamma_fixed_zero:
        model_label = (
            "naive_feedback_gamma_zero"
        )

    elif independent_brownian:
        model_label = (
            "naive_feedback_independent_brownian"
        )

    else:
        model_label = (
            "naive_feedback_mean_reverting_z_sde"
        )

    model_path = CHECKPOINT_DIR / (
        f"{model_label}_{transform}_{loss_type}_{ticker}.pth"
    )

    loss_path = CHECKPOINT_DIR / (
        f"{model_label}_{transform}_{loss_type}_{ticker}_loss.npy"
    )

    current_epoch = 0
    best_epoch = -1
    best_loss = float("inf")
    losses: List[float] = []

    if resume and model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if initial_state_dict is not None:
            model.load_state_dict(initial_state_dict)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint["loss"])
        target_stats = checkpoint.get("target_stats", target_stats)
        if loss_path.exists():
            losses = np.load(loss_path).astype(float).tolist()
        print(
            f"Loaded {model_path.name} from epoch {best_epoch}; "
            f"best loss={best_loss:.6f}"
        )

    total_time = float(forecast_horizon)
    n_steps_total = n_substeps * forecast_horizon
    ts = torch.linspace(
        0.0,
        total_time,
        n_steps_total + 1,
        dtype=torch.float32,
        device=device,
    )
    dt = total_time / n_steps_total

    n_candidates = len(Y_obs) - forecast_horizon

    for epoch in range(current_epoch, n_epochs):
        model.train()

        replace = n_candidates < batch_size
        indices = np.random.choice(
            n_candidates,
            size=batch_size,
            replace=replace,
        )
        future_indices = indices + forecast_horizon

        Y0 = torch.as_tensor(
            Y_obs[indices], dtype=torch.float32, device=device
        ).view(-1, 1)
        R0 = torch.as_tensor(
            R_obs[indices], dtype=torch.float32, device=device
        ).view(-1, 1)

        Y1_true = torch.as_tensor(
            Y_obs[future_indices], dtype=torch.float32, device=device
        ).view(-1, 1)
        R1_true = torch.as_tensor(
            R_obs[future_indices], dtype=torch.float32, device=device
        ).view(-1, 1)

        X0 = initialise_state(model, Y0, R0, init_mode)

        _, return_particles, log_R_particles, H_particles = simulate_particles(
            model=model,
            X0=X0,
            ts=ts,
            n_particles=n_particles,
            dt=dt,
        )

        return_true = 100.0 * (Y1_true - Y0).squeeze(-1)
        log_R_true = torch.log(R1_true.squeeze(-1).clamp_min(1e-8))

        if loss_type in ("joint_qnll", "composite_nll"):
            return_particles_std, log_R_particles_std = standardise_joint_targets(
                return_particles, log_R_particles, target_stats
            )
            return_true_std, log_R_true_std = standardise_joint_targets(
                return_true, log_R_true, target_stats
            )

        if loss_type == "joint_qnll":
            loss = joint_gaussian_qnll(
                return_particles=return_particles_std,
                log_rv_particles=log_R_particles_std,
                return_true=return_true_std,
                log_rv_true=log_R_true_std,
                covariance_jitter=covariance_jitter,
            )
        elif loss_type == "composite_nll":
            loss = composite_predictive_nll(
                return_particles=return_particles_std,
                log_rv_particles=log_R_particles_std,
                return_true=return_true_std,
                log_rv_true=log_R_true_std,
                lambda_rv=lambda_rv,
            )
        elif loss_type == "return_qnll":
            H_pred = H_particles.mean(dim=0)
            loss = return_qnll(H_pred, return_true)

        elif loss_type == "return_rv":
            H_pred = H_particles.mean(dim=0)

            loss, return_nll_train, rv_qlike_train = return_rv_loss(
                predicted_integrated_variance=H_pred,
                return_true=return_true,
                rv_true=R1_true.squeeze(-1),
                lambda_rv=lambda_rv,
                return_components=True,
            )

        else:
            raise ValueError(f"Unknown loss_type={loss_type}")

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}. "
                "Try a smaller learning rate, more covariance jitter, or more particles."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        current_loss = float(loss.detach().cpu().item())
        losses.append(current_loss)
        np.save(loss_path, np.asarray(losses, dtype=np.float32))

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "loss": best_loss,

                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "target_stats": target_stats,
                    "transform": transform,
                    "loss_type": loss_type,
                    "forecast_horizon": forecast_horizon,
                    "n_substeps": n_substeps,
                    "gamma_fixed_zero": gamma_fixed_zero,
                    "independent_brownian": independent_brownian,
                    "feedback_specification": "naive",
                },
                model_path,
            )

        if epoch % 50 == 0:
            with torch.no_grad():
                H_mean = float(H_particles.mean().cpu().item())
                R_pred_mean = float(torch.exp(log_R_particles).mean().cpu().item())
                r2_mean = float(return_true.square().mean().cpu().item())

            if loss_type == "return_rv":
                print(
                    f"Epoch [{epoch}/{n_epochs}] | "
                    f"loss={current_loss:.6f} | "
                    f"return_NLL={return_nll_train.item():.6f} | "
                    f"RV_QLIKE={rv_qlike_train.item():.6f} | "
                    f"weighted_RV={lambda_rv * rv_qlike_train.item():.6f} | "
                    f"mean(r^2)={r2_mean:.6f} | "
                    f"mean(H)={H_mean:.6f}"
        )
            print(
                f"Epoch [{epoch}/{n_epochs}] | "
                f"mean(pred R)={R_pred_mean:.6f} | "
                f"kappa={model.kappa.item():.4f} | "
                f"kappa_z={model.kappa_z.item():.4f} | "
                f"theta={model.theta.item():.4f} | "
                f"gamma={model.gamma.item():.4f} | "
                f"xi={model.xi.item():.4f} | "
                f"phi={model.phi.item():.4f} | "
                f"gamma*phi={(model.gamma * model.phi).item():.4f} | "
                f"kappa-gamma*phi="
                f"{(model.kappa - model.gamma * model.phi).item():.4f} | "
                f"eta_s={model.eta_s.item():.4f} | "
                f"eta_R={model.eta_R.item():.4f} | "
                f"sigma_u={model.sigma_u.item():.4f} | "
                f"rho_Ys={model.rho_Ys.item():.4f} | "
                f"rho_YR={model.rho_YR.item():.4f}"
            )

    if best_epoch < 0:
        raise RuntimeError("No checkpoint was saved during training.")

    best_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(
        f"Training complete. Best epoch={best_epoch}, "
        f"best loss={best_loss:.6f}"
    )
    return model, losses


# =========================================================
# Direct one-step/multi-step forecasting
# =========================================================
@torch.no_grad()
def direct_forecast(
    ticker: str,
    log_prices_test: np.ndarray,
    rv_test: np.ndarray,
    init_mode: str = "from_rv",
    transform: TransformType = "exp",
    loss_type: LossType = "return_rv",
    forecast_horizon: int = 1,
    n_particles: int = 256,
    n_substeps: int = 20,
    device: Optional[torch.device] = None,
    gamma_fixed_zero: bool = False,
    independent_brownian: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_prices_test = np.asarray(log_prices_test, dtype=np.float64).reshape(-1)
    rv_test = np.maximum(np.asarray(rv_test, dtype=np.float64).reshape(-1), 1e-8)

    if len(log_prices_test) != len(rv_test):
        raise ValueError("log_prices_test and rv_test must have the same length.")
    if len(log_prices_test) <= forecast_horizon:
        raise ValueError("Test series is too short for the requested horizon.")

    model = PositiveJointSDE(
        transform=transform,
        gamma_fixed_zero=gamma_fixed_zero,
        independent_brownian=independent_brownian,
    ).to(device)

    if gamma_fixed_zero:
        model_label = (
            "naive_feedback_gamma_zero"
        )

    elif independent_brownian:
        model_label = (
            "naive_feedback_independent_brownian"
        )

    else:
        model_label = (
            "naive_feedback_mean_reverting_z_sde"
        )

    model_path = CHECKPOINT_DIR / (
        f"{model_label}_{transform}_{loss_type}_{ticker}.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_time = float(forecast_horizon)
    n_steps_total = n_substeps * forecast_horizon
    ts = torch.linspace(
        0.0,
        total_time,
        n_steps_total + 1,
        dtype=torch.float32,
        device=device,
    )
    dt = total_time / n_steps_total

    predicted_log_price: List[float] = []
    predicted_H: List[float] = []
    predicted_R: List[float] = []

    for i in range(len(log_prices_test) - forecast_horizon):
        Y0 = torch.tensor(
            [[log_prices_test[i]]], dtype=torch.float32, device=device
        )
        R0 = torch.tensor(
            [[rv_test[i]]], dtype=torch.float32, device=device
        )

        X0 = initialise_state(model, Y0, R0, init_mode)
        path, _, log_R_particles, H_particles = simulate_particles(
            model=model,
            X0=X0,
            ts=ts,
            n_particles=n_particles,
            dt=dt,
        )

        Y1_mean = path[-1, :, 0, 0].mean()
        H_mean = H_particles[:, 0].mean()
        R_mean = torch.exp(log_R_particles[:, 0]).mean()

        predicted_log_price.append(float(Y1_mean.cpu().item()))
        predicted_H.append(float(H_mean.cpu().item()))
        predicted_R.append(float(R_mean.cpu().item()))

    actual_next_log_price = log_prices_test[forecast_horizon:]
    actual_percentage_return = 100.0 * (
        log_prices_test[forecast_horizon:]
        - log_prices_test[:-forecast_horizon]
    )
    actual_next_rv = rv_test[forecast_horizon:]

    return (
        np.asarray(predicted_log_price),
        np.asarray(predicted_H),
        np.asarray(predicted_R),
        np.asarray(actual_next_log_price),
        np.asarray(actual_percentage_return),
        np.asarray(actual_next_rv),
    )


# =========================================================
# Evaluation metrics
# =========================================================
def qlike_var_np(
    predicted_variance: np.ndarray,
    realised_variance: np.ndarray,
    eps: float = 1e-8,
) -> float:
    h = np.maximum(np.asarray(predicted_variance, dtype=np.float64), eps)
    rv = np.maximum(np.asarray(realised_variance, dtype=np.float64), eps)
    return float(np.mean(np.log(h) + rv / h))


def mse_vol_np(
    predicted_variance: np.ndarray,
    realised_variance: np.ndarray,
    eps: float = 1e-12,
) -> float:
    h = np.maximum(np.asarray(predicted_variance, dtype=np.float64), eps)
    rv = np.maximum(np.asarray(realised_variance, dtype=np.float64), eps)
    return float(np.mean((np.sqrt(rv) - np.sqrt(h)) ** 2))


def mse_var_np(
    predicted_variance: np.ndarray,
    realised_variance: np.ndarray,
    eps: float = 1e-12,
) -> float:
    h = np.maximum(np.asarray(predicted_variance, dtype=np.float64), eps)
    rv = np.maximum(np.asarray(realised_variance, dtype=np.float64), eps)
    return float(np.mean((rv - h) ** 2))

def mae_var_np(
    predicted_variance: np.ndarray,
    realised_variance: np.ndarray,
) -> float:
    """
    Mean Absolute Error (variance scale).
    """
    pred = np.asarray(predicted_variance, dtype=np.float64).reshape(-1)
    true = np.asarray(realised_variance, dtype=np.float64).reshape(-1)

    if len(pred) != len(true):
        raise ValueError(
            "predicted_variance and realised_variance must have equal length."
        )

    return float(np.mean(np.abs(pred - true)))

def mae_vol_np(
    predicted_variance: np.ndarray,
    realised_variance: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Mean Absolute Error (volatility scale).
    """
    pred_vol = np.sqrt(
        np.maximum(
            np.asarray(predicted_variance, dtype=np.float64).reshape(-1),
            eps,
        )
    )

    true_vol = np.sqrt(
        np.maximum(
            np.asarray(realised_variance, dtype=np.float64).reshape(-1),
            eps,
        )
    )

    if len(pred_vol) != len(true_vol):
        raise ValueError(
            "predicted_variance and realised_variance must have equal length."
        )

    return float(np.mean(np.abs(pred_vol - true_vol)))

def return_nll_np(
    predicted_variance: np.ndarray,
    returns_true: np.ndarray,
    eps: float = 1e-8,
) -> float:
    h = np.maximum(np.asarray(predicted_variance, dtype=np.float64).reshape(-1), eps)
    r = np.asarray(returns_true, dtype=np.float64).reshape(-1)

    if len(h) != len(r):
        raise ValueError("predicted_variance and returns_true must have equal length.")

    nll = 0.5 * (
        np.log(2.0 * np.pi) + np.log(h) + r ** 2 / h
    )
    return float(np.mean(nll))

# =========================================================
# Plotting
# =========================================================
def save_variance_plot(
    ticker: str,
    actual_rv: np.ndarray,
    predicted_variance: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_rv, label="Actual RV", linewidth=1.5)
    ax.plot(predicted_variance, label="Predicted integrated variance", linewidth=1.0)
    ax.set_title(f"{ticker}: predicted integrated variance vs actual RV")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{ticker}_positive_sde_variance_fit.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_realised_state_plot(
    ticker: str,
    actual_rv: np.ndarray,
    predicted_rv_state: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_rv, label="Actual RV", linewidth=1.5)
    ax.plot(predicted_rv_state, label="Predicted R state", linewidth=1.0)
    ax.set_title(f"{ticker}: continuous-time realised-measure state")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Realised-measure level")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{ticker}_positive_sde_realised_state_fit.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# =========================================================
# Rolling-window forecasting
# =========================================================
@torch.no_grad()
def forecast_one_step_with_model(
    model: PositiveJointSDE,
    Y0_value: float,
    R0_value: float,
    init_mode: str = "from_rv",
    n_particles: int = 256,
    n_substeps: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float]:
    """Forecast t+1 from observed information at t using an in-memory model."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ts = torch.linspace(
        0.0, 1.0, n_substeps + 1,
        dtype=torch.float32, device=device,
    )
    dt = 1.0 / n_substeps

    Y0 = torch.tensor([[Y0_value]], dtype=torch.float32, device=device)
    R0 = torch.tensor([[max(R0_value, 1e-8)]], dtype=torch.float32, device=device)
    X0 = initialise_state(model, Y0, R0, init_mode)

    path, _, log_R_particles, H_particles = simulate_particles(
        model=model,
        X0=X0,
        ts=ts,
        n_particles=n_particles,
        dt=dt,
    )

    predicted_log_price = path[-1, :, 0, 0].mean().item()
    predicted_H = H_particles[:, 0].mean().item()
    predicted_R = torch.exp(log_R_particles[:, 0]).mean().item()
    return predicted_log_price, predicted_H, predicted_R


# ============================================================
# Fixed-window rolling SDE forecasting
# ============================================================

def rolling_window_forecast(
    ticker: str,
    y_train: np.ndarray,
    rv_train: np.ndarray,
    y_test: np.ndarray,
    rv_test: np.ndarray,
    window_size: Optional[int] = None,
    refit_every: int = 21,
    init_mode: str = "from_rv",
    transform: TransformType = "exp",
    loss_type: LossType = "return_rv",
    n_substeps_train: int = 10,
    n_particles_train: int = 64,
    n_epochs_initial: int = 1000,
    n_epochs_refit: int = 300,
    lr: float = 1e-4,
    batch_size: int = 32,
    lambda_rv: float = 0.1,
    covariance_jitter: float = 1e-4,
    grad_clip: float = 5.0,
    n_particles_forecast: int = 256,
    n_substeps_forecast: int = 20,
    device: Optional[torch.device] = None,
    gamma_fixed_zero: bool = False,
    independent_brownian: bool = False,
) -> Tuple[

    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
]:
    """
    Fixed-length rolling-window SDE evaluation.

    Design
    ------
    1. A fixed estimation window of length `window_size` is used.
    2. Parameters are re-estimated every `refit_every` trading days.
    3. The estimation window advances ONLY on refit dates.
    4. One-step-ahead forecasts are generated EVERY trading day.
    5. Between refits, SDE parameters remain fixed.
    6. Each daily forecast is initialized from the observed
       state at the previous trading day.

    Example with window_size=50 and refit_every=21:

        Fit on [0, 49]
        Forecast targets 50, ..., 70

        Fit on [21, 70]
        Forecast targets 71, ..., 91

        Fit on [42, 91]
        Forecast targets 92, ..., 112

    This matches the block-rolling logic used by the
    Realized-GARCH benchmark.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if refit_every < 1:
        raise ValueError(
            "refit_every must be at least 1."
        )

    # ========================================================
    # Arrays
    # ========================================================

    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(-1)

    rv_train = np.maximum(
        np.asarray(
            rv_train,
            dtype=np.float64,
        ).reshape(-1),
        1e-8,
    )

    y_test = np.asarray(
        y_test,
        dtype=np.float64,
    ).reshape(-1)

    rv_test = np.maximum(
        np.asarray(
            rv_test,
            dtype=np.float64,
        ).reshape(-1),
        1e-8,
    )

    if (
        len(y_train) != len(rv_train)
        or len(y_test) != len(rv_test)
    ):
        raise ValueError(
            "Each price array must match its RV array."
        )

    if window_size is None:
        window_size = len(y_train)

    if (
        window_size < 2
        or window_size > len(y_train)
    ):
        raise ValueError(
            "window_size must lie between 2 and len(y_train)."
        )

    # ========================================================
    # Full chronological sample
    # ========================================================

    full_y = np.concatenate(
        [
            y_train,
            y_test,
        ]
    )

    full_rv = np.concatenate(
        [
            rv_train,
            rv_test,
        ]
    )

    first_oos_index = len(
        y_train
    )

    n_total = len(
        full_y
    )

    n_oos = (
        n_total
        - first_oos_index
    )

    # ========================================================
    # Storage
    # ========================================================

    predicted_log_price: List[float] = []
    predicted_H: List[float] = []
    predicted_R: List[float] = []

    actual_log_price: List[float] = []
    actual_return: List[float] = []
    actual_rv: List[float] = []

    fitted_model: Optional[
        PositiveJointSDE
    ] = None

    # --------------------------------------------------------
    # IMPORTANT:
    # The active estimation window changes only when parameters
    # are re-estimated.
    # --------------------------------------------------------

    active_window_start: Optional[int] = None
    active_window_end: Optional[int] = None

    temporary_ticker = (
        f"{ticker}_rolling_tmp"
    )

    parameter_history: Dict[
        str,
        List
    ] = {
        "forecast_number": [],
        "target_index": [],
        "window_start": [],
        "window_end": [],
        "refit": [],

        "kappa": [],
        "kappa_z": [],
        "theta": [],
        "gamma": [],
        "xi": [],
        "phi": [],

        "gamma_phi": [],
        "stability_margin": [],

        "eta_s": [],
        "eta_R": [],
        "sigma_u": [],

        "rho_Ys": [],
        "rho_YR": [],

        "mean_reversion_drift": [],
        "feedback_drift": [],
        "measurement_reversion_drift": [],
    }

    # ========================================================
    # Diagnostics
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        f"{ticker}: block rolling SDE"
    )

    print(
        "=" * 90
    )

    print(
        f"Total observations       : "
        f"{n_total}"
    )

    print(
        f"Estimation-window size   : "
        f"{window_size}"
    )

    print(
        f"Out-of-sample forecasts  : "
        f"{n_oos}"
    )

    print(
        f"Parameter refit frequency: "
        f"every {refit_every} trading days"
    )

    print(
        "Forecast frequency       : daily"
    )

    # ========================================================
    # Daily OOS loop
    # ========================================================

    for forecast_number, target_idx in enumerate(
        range(
            first_oos_index,
            n_total,
        ),
        start=1,
    ):

        # ----------------------------------------------------
        # Same convention as Realized GARCH:
        #
        # forecast 1, 22, 43, ...
        #
        # for refit_every = 21.
        # ----------------------------------------------------

        should_refit = (
            fitted_model is None
            or (
                forecast_number - 1
            )
            % refit_every
            == 0
        )

        # ====================================================
        # Parameter re-estimation
        # ====================================================

        if should_refit:

            # Window ends immediately before current target.
            active_window_end = (
                target_idx
            )

            active_window_start = (
                active_window_end
                - window_size
            )

            if active_window_start < 0:
                raise RuntimeError(
                    "Active estimation window begins "
                    "before the sample."
                )

            y_window = full_y[
                active_window_start:
                active_window_end
            ]

            rv_window = full_rv[
                active_window_start:
                active_window_end
            ]

            if len(
                y_window
            ) != window_size:
                raise RuntimeError(
                    f"Unexpected SDE window length: "
                    f"{len(y_window)} instead of "
                    f"{window_size}."
                )

            if len(
                y_window
            ) != len(
                rv_window
            ):
                raise RuntimeError(
                    "SDE price and RV windows "
                    "are not aligned."
                )

            # ------------------------------------------------
            # First fit can use more epochs.
            # Later monthly refits use n_epochs_refit.
            # ------------------------------------------------

            epochs = (
                n_epochs_initial
                if fitted_model is None
                else n_epochs_refit
            )

            # ------------------------------------------------
            # Train ONLY on current active rolling window.
            #
            # resume=False prevents accidentally loading an
            # old checkpoint containing observations outside
            # this window.
            # ------------------------------------------------
            previous_state_dict = None

            if fitted_model is not None:

                previous_state_dict = {
                    key: value.detach().clone()
                    for key, value
                    in fitted_model.state_dict().items()
                }

            fitted_model, _ = train(
                ticker=temporary_ticker,
                Y_obs=y_window,
                R_obs=rv_window,
                init_mode=init_mode,
                transform=transform,
                loss_type=loss_type,
                forecast_horizon=1,
                n_substeps=n_substeps_train,
                n_particles=n_particles_train,
                n_epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                lambda_rv=lambda_rv,
                covariance_jitter=(
                    covariance_jitter
                ),
                grad_clip=grad_clip,
                device=device,
                resume=False,
                initial_state_dict=previous_state_dict,
                gamma_fixed_zero=gamma_fixed_zero,
                independent_brownian=independent_brownian,
            )

            print(
                f"\nRefit {forecast_number} | "
                f"window=["
                f"{active_window_start}, "
                f"{active_window_end - 1}]"
            )

        # ====================================================
        # Safety checks
        # ====================================================

        if (
            fitted_model is None
            or active_window_start is None
            or active_window_end is None
        ):
            raise RuntimeError(
                "Rolling SDE has not been initialized."
            )

        # ====================================================
        # One-step-ahead forecast
        # ====================================================
        #
        # For target t:
        #
        #     origin = t - 1
        #
        # Use information observed at t-1.
        #
        # Parameters remain fixed between refits.
        # ====================================================

        origin_idx = (
            target_idx - 1
        )

        (
            pred_y,
            pred_h,
            pred_r,
        ) = forecast_one_step_with_model(
            model=fitted_model,
            Y0_value=full_y[
                origin_idx
            ],
            R0_value=full_rv[
                origin_idx
            ],
            init_mode=init_mode,
            n_particles=(
                n_particles_forecast
            ),
            n_substeps=(
                n_substeps_forecast
            ),
            device=device,
        )

        # ====================================================
        # Current fitted parameters
        # ====================================================

        with torch.no_grad():
            current_kappa = float(
                fitted_model.kappa.detach().cpu().item()
            )

            current_kappa_z = float(
                fitted_model.kappa_z.detach().cpu().item()
            )

            current_theta = float(
                fitted_model.theta.detach().cpu().item()
            )

            current_gamma = float(
                fitted_model.gamma.detach().cpu().item()
            )

            current_xi = float(
                fitted_model.xi.detach().cpu().item()
            )

            current_phi = float(
                fitted_model.phi.detach().cpu().item()
            )

            current_eta_s = float(
                fitted_model.eta_s.detach().cpu().item()
            )

            current_eta_R = float(
                fitted_model.eta_R.detach().cpu().item()
            )

            current_sigma_u = float(
                fitted_model.sigma_u.detach().cpu().item()
            )

            current_rho_Ys = float(
                fitted_model.rho_Ys.detach().cpu().item()
            )

            current_rho_YR = float(
                fitted_model.rho_YR.detach().cpu().item()
            )

            R0_value = float(
                full_rv[origin_idx]
            )

            R0_tensor = torch.tensor(
                [[max(R0_value, 1e-8)]],
                dtype=torch.float32,
                device=device,
            )

            if init_mode == "from_rv":

                x_origin = (
                    fitted_model
                    .inverse_variance_transform(
                        R0_tensor
                    )
                    .squeeze()
                )

            elif init_mode == "long_run":

                x_origin = (
                    fitted_model.theta
                )

            else:

                raise ValueError(
                    "init_mode must be "
                    "'from_rv' or 'long_run'."
                )

            z_origin = (
                fitted_model
                .inverse_realised_transform(
                    R0_tensor
                )
                .squeeze()
            )

            # -----------------------------------------------
            # dx drift decomposition
            #
            # dx =
            # kappa(theta-x)
            # + gamma*z
            # -----------------------------------------------

            current_mean_reversion_drift = float(
                (
                    fitted_model.kappa
                    * (
                        fitted_model.theta
                        - x_origin
                    )
                )
                .detach()
                .cpu()
                .item()
            )

            # -----------------------------------------------
            # Naive feedback drift:
            # gamma * z
            # -----------------------------------------------

            current_feedback_drift = float(
                (
                    fitted_model.gamma
                    * z_origin
                )
                .detach()
                .cpu()
                .item()
            )

            current_measurement_reversion_drift = float(
                (
                    fitted_model.kappa_z
                    * (
                        fitted_model.xi
                        + fitted_model.phi * x_origin
                        - z_origin
                    )
                )
                .detach()
                .cpu()
                .item()
            )

            current_gamma_phi = float(
                (
                    fitted_model.gamma
                    * fitted_model.phi
                )
                .detach()
                .cpu()
                .item()
            )

            current_stability_margin = float(
                (
                    fitted_model.kappa
                    - fitted_model.gamma
                    * fitted_model.phi
                )
                .detach()
                .cpu()
                .item()
            )
        # ====================================================
        # Save forecast
        # ====================================================

        predicted_log_price.append(
            pred_y
        )

        predicted_H.append(
            pred_h
        )

        predicted_R.append(
            pred_r
        )

        actual_log_price.append(
            float(
                full_y[
                    target_idx
                ]
            )
        )

        actual_return.append(
            float(
                100.0
                * (
                    full_y[
                        target_idx
                    ]
                    - full_y[
                        origin_idx
                    ]
                )
            )
        )

        actual_rv.append(
            float(
                full_rv[
                    target_idx
                ]
            )
        )

        # ====================================================
        # Save parameter history
        # ====================================================

        parameter_history[
            "forecast_number"
        ].append(
            forecast_number
        )

        parameter_history[
            "target_index"
        ].append(
            target_idx
        )

        parameter_history[
            "window_start"
        ].append(
            active_window_start
        )

        parameter_history[
            "window_end"
        ].append(
            active_window_end - 1
        )

        parameter_history[
            "refit"
        ].append(
            bool(
                should_refit
            )
        )

        parameter_history[
            "kappa"
        ].append(
            current_kappa
        )

        parameter_history[
            "kappa_z"
        ].append(
            current_kappa_z
        )

        parameter_history[
            "theta"
        ].append(
            current_theta
        )

        parameter_history[
            "gamma"
        ].append(
            current_gamma
        )

        parameter_history[
            "xi"
        ].append(
            current_xi
        )

        parameter_history[
            "phi"
        ].append(
            current_phi
        )

        parameter_history[
            "eta_s"
        ].append(
            current_eta_s
        )

        parameter_history[
            "eta_R"
        ].append(
            current_eta_R
        )

        parameter_history[
            "sigma_u"
        ].append(
            current_sigma_u
        )

        parameter_history[
            "rho_Ys"
        ].append(
            current_rho_Ys
        )

        parameter_history[
            "rho_YR"
        ].append(
            current_rho_YR
        )

        parameter_history[
            "mean_reversion_drift"
        ].append(
            current_mean_reversion_drift
        )

        parameter_history[
            "feedback_drift"
        ].append(
            current_feedback_drift
        )

        parameter_history[
            "measurement_reversion_drift"
        ].append(
            current_measurement_reversion_drift
        )

        parameter_history[
            "gamma_phi"
        ].append(
            current_gamma_phi
        )

        parameter_history[
            "stability_margin"
        ].append(
            current_stability_margin
        )

        # ====================================================
        # Progress
        # ====================================================

        print(
            f"{ticker} forecast "
            f"{forecast_number}/{n_oos} | "
            f"estimation_window="
            f"[{active_window_start}, "
            f"{active_window_end - 1}] | "
            f"target={target_idx} | "
            f"refit={should_refit} | "
            f"H={pred_h:.6f}"
        )

    # ========================================================
    # Convert parameter history
    # ========================================================

    parameter_history_np = {
        key: np.asarray(
            values
        )
        for key, values
        in parameter_history.items()
    }

    # ========================================================
    # Return
    # ========================================================

    return (
        np.asarray(
            predicted_log_price,
            dtype=np.float64,
        ),

        np.asarray(
            predicted_H,
            dtype=np.float64,
        ),

        np.asarray(
            predicted_R,
            dtype=np.float64,
        ),

        np.asarray(
            actual_log_price,
            dtype=np.float64,
        ),

        np.asarray(
            actual_return,
            dtype=np.float64,
        ),

        np.asarray(
            actual_rv,
            dtype=np.float64,
        ),

        parameter_history_np,
    )

# ============================================================
# Fixed 50:50 block rolling SDE
# ============================================================

def fixed_half_sample_rolling_forecast(
    ticker: str,
    y_full: np.ndarray,
    rv_full: np.ndarray,
    refit_every: int = 21,
    init_mode: str = "from_rv",
    transform: TransformType = "exp",
    loss_type: LossType = "return_rv",
    n_substeps_train: int = 10,
    n_particles_train: int = 64,
    n_epochs_initial: int = 1000,
    n_epochs_refit: int = 300,
    lr: float = 1e-4,
    batch_size: int = 32,
    lambda_rv: float = 0.1,
    covariance_jitter: float = 1e-4,
    grad_clip: float = 5.0,
    n_particles_forecast: int = 256,
    n_substeps_forecast: int = 20,
    device: Optional[torch.device] = None,
    gamma_fixed_zero: bool = False,
    independent_brownian: bool = False,
):
    """
    Fixed 50:50 block rolling-window SDE evaluation.

    Design
    ------
    1. window_size = floor(N / 2)
    2. first OOS observation = window_size
    3. parameters refitted every 21 trading days
    4. estimation window advances 21 observations per refit
    5. one-step forecasts generated every trading day

    Example:

        initial fit:
            [0, ..., W-1]

        forecasts:
            W, ..., W+20

        second fit:
            [21, ..., W+20]

        forecasts:
            W+21, ..., W+41

        third fit:
            [42, ..., W+41]

        etc.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    y_full = np.asarray(
        y_full,
        dtype=np.float64,
    ).reshape(-1)

    rv_full = np.maximum(
        np.asarray(
            rv_full,
            dtype=np.float64,
        ).reshape(-1),
        1e-8,
    )

    if len(
        y_full
    ) != len(
        rv_full
    ):
        raise ValueError(
            "y_full and rv_full must "
            "have the same length."
        )

    if len(
        y_full
    ) < 4:
        raise ValueError(
            "At least four observations "
            "are required."
        )

    if not np.all(
        np.isfinite(
            y_full
        )
    ):
        raise ValueError(
            "y_full contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            rv_full
        )
    ):
        raise ValueError(
            "rv_full contains non-finite values."
        )

    if refit_every < 1:
        raise ValueError(
            "refit_every must be at least 1."
        )

    # ========================================================
    # Exactly the same 50:50 definition as baseline
    # ========================================================

    n_total = len(
        y_full
    )

    window_size = (
        n_total // 2
    )

    first_oos_index = (
        window_size
    )

    n_oos = (
        n_total
        - first_oos_index
    )

    y_initial_train = (
        y_full[
            :window_size
        ]
    )

    rv_initial_train = (
        rv_full[
            :window_size
        ]
    )

    y_oos = (
        y_full[
            first_oos_index:
        ]
    )

    rv_oos = (
        rv_full[
            first_oos_index:
        ]
    )

    print(
        "\n" + "=" * 90
    )

    print(
        f"{ticker}: fixed 50:50 "
        "block rolling SDE"
    )

    print(
        "=" * 90
    )

    print(
        f"Total observations       : "
        f"{n_total}"
    )

    print(
        f"Estimation-window size   : "
        f"{window_size}"
    )

    print(
        f"Out-of-sample forecasts  : "
        f"{n_oos}"
    )

    print(
        f"Parameter refit frequency: "
        f"every {refit_every} "
        "trading days"
    )

    print(
        "Forecast frequency       : daily"
    )

    return rolling_window_forecast(
        ticker=ticker,

        y_train=
            y_initial_train,

        rv_train=
            rv_initial_train,

        y_test=
            y_oos,

        rv_test=
            rv_oos,

        window_size=
            window_size,

        refit_every=
            refit_every,

        init_mode=
            init_mode,

        transform=
            transform,

        loss_type=
            loss_type,

        n_substeps_train=
            n_substeps_train,

        n_particles_train=
            n_particles_train,

        n_epochs_initial=
            n_epochs_initial,

        n_epochs_refit=
            n_epochs_refit,

        lr=
            lr,

        batch_size=
            batch_size,

        lambda_rv=
            lambda_rv,

        covariance_jitter=
            covariance_jitter,

        grad_clip=
            grad_clip,

        n_particles_forecast=
            n_particles_forecast,

        n_substeps_forecast=
            n_substeps_forecast,

        device=
            device,

        gamma_fixed_zero=
            gamma_fixed_zero,

        independent_brownian=independent_brownian,
    )

def save_prediction_scatter_plot(
    ticker: str,
    actual_rv: np.ndarray,
    predicted_variance: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
    prediction_label: str = "Predicted integrated variance",
    use_log_scale: bool = False,
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = np.asarray(
        actual_rv,
        dtype=np.float64,
    ).reshape(-1)

    predicted = np.asarray(
        predicted_variance,
        dtype=np.float64,
    ).reshape(-1)

    if len(actual) != len(predicted):
        raise ValueError(
            "actual_rv and predicted_variance must have equal length."
        )

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (actual > 0.0)
        & (predicted > 0.0)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError(
            "No valid positive observations are available for plotting."
        )

    lower = min(
        float(np.min(actual)),
        float(np.min(predicted)),
    )

    upper = max(
        float(np.max(actual)),
        float(np.max(predicted)),
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    ax.scatter(
        actual,
        predicted,
        alpha=0.55,
        s=18,
        edgecolors="none",
    )

    # Perfect-prediction line.
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
        label="45-degree line",
    )

    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Observed realised variance")
    ax.set_ylabel(prediction_label)

    scale_text = "log scale" if use_log_scale else "level scale"

    ax.set_title(
        f"{ticker}: predicted versus observed variance ({scale_text})"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    suffix = "log" if use_log_scale else "level"

    fig.savefig(
        output_dir
        / f"{ticker}_predicted_vs_observed_scatter_{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# Rolling-correlation evolution plot
# =========================================================
def save_rolling_correlation_plot(
    ticker: str,
    parameter_history: Dict[str, np.ndarray],
    output_dir: Path = OUTPUT_DIR,
    plot_only_refits: bool = True,
) -> None:
    """
    Plot the rolling evolution of rho_Ys and rho_YR.

    If plot_only_refits=True, only parameter re-estimation dates are shown.
    This avoids long flat sections when refit_every > 1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_number = np.asarray(
        parameter_history["forecast_number"]
    ).reshape(-1)

    rho_Ys = np.asarray(
        parameter_history["rho_Ys"],
        dtype=np.float64,
    ).reshape(-1)

    rho_YR = np.asarray(
        parameter_history["rho_YR"],
        dtype=np.float64,
    ).reshape(-1)

    refit = np.asarray(
        parameter_history["refit"],
        dtype=bool,
    ).reshape(-1)

    if not (
        len(forecast_number)
        == len(rho_Ys)
        == len(rho_YR)
        == len(refit)
    ):
        raise ValueError(
            "Parameter-history arrays must have equal length."
        )

    if plot_only_refits:
        mask = refit
        x = forecast_number[mask]
        ys = rho_Ys[mask]
        yr = rho_YR[mask]
        x_label = "Rolling refit forecast number"
        suffix = "refits_only"
    else:
        x = forecast_number
        ys = rho_Ys
        yr = rho_YR
        x_label = "Out-of-sample forecast number"
        suffix = "daily"

    if len(x) == 0:
        raise ValueError(
            "No parameter observations are available for plotting."
        )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        x,
        ys,
        linewidth=1.5,
        label=r"$\rho_{Ys}$",
    )

    ax.plot(
        x,
        yr,
        linewidth=1.5,
        label=r"$\rho_{YR}$",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Estimated correlation")
    ax.set_title(
        f"{ticker}: rolling innovation-correlation estimates"
    )

    ax.set_ylim(-1.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{ticker}_rolling_correlations_{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
