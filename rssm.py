import torch
import torchode
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        if deter % self.blocks != 0:
            raise ValueError(f"deter ({deter}) must be divisible by blocks ({self.blocks}).")
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)

    def _flat_to_group(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.blocks, -1)

    @staticmethod
    def _group_to_flat(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-2], -1)

    def _candidate_and_update(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the original block-GRU candidate and update gate."""
        # (B, S, K), (B, D), (B, A)
        batch_size = action.shape[0]

        # Flatten stochastic state and normalize action magnitude.
        # (B, S*K)
        stoch = stoch.reshape(batch_size, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # (B, U)
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        # Concatenate projected inputs and broadcast over blocks.
        # (B, 3*U)
        x = torch.cat([x0, x1, x2], -1)
        # (B, G, 3*U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Combine per-block deterministic state with per-block inputs.
        # (B, G, D/G + 3*U) -> (B, D + 3*U*G)
        x = self._group_to_flat(torch.cat([self._flat_to_group(deter), x], -1))

        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D)
        x = self._dyn_gru(x)

        # Split GRU-style gates block-wise.
        # (B, G, 3*D/G)
        gates = torch.chunk(self._flat_to_group(x), 3, dim=-1)

        # (B, D)
        reset, cand, update = (self._group_to_flat(value) for value in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return cand, update

    def forward(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic state transition (block-GRU style)."""
        cand, update = self._candidate_and_update(stoch, deter, action)
        # (B, D)
        return update * cand + (1 - update) * deter


def linear_interval_velocity_loss(
    predicted_velocity: torch.Tensor,
    embeddings: torch.Tensor,
    delta_t: torch.Tensor | None = None,
    reset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match projected vector-field values to detached linear embedding slopes."""
    batch_size, sequence_length, embed_size = embeddings.shape

    if delta_t is None:
        duration = torch.ones(
            batch_size,
            sequence_length,
            device=embeddings.device,
            dtype=torch.float32,
        )
    else:
        duration = torch.as_tensor(delta_t, device=embeddings.device, dtype=torch.float32).reshape(
            batch_size, sequence_length
        )

    if reset is None:
        reset_mask = torch.zeros(
            batch_size,
            sequence_length,
            device=embeddings.device,
            dtype=torch.bool,
        )
    else:
        reset_mask = torch.as_tensor(reset, device=embeddings.device, dtype=torch.bool).reshape(
            batch_size, sequence_length
        )

    interval_duration = duration[:, 1:]
    valid = (interval_duration > 0.0) & ~reset_mask[:, 1:]
    safe_duration = torch.where(valid, interval_duration, torch.ones_like(interval_duration))
    target_velocity = (embeddings[:, 1:].float() - embeddings[:, :-1].float()) / safe_duration.unsqueeze(-1)
    squared_error = (predicted_velocity.float() - target_velocity.detach()).square()
    valid_elements = valid.sum().to(squared_error.dtype) * embed_size
    return (squared_error * valid.unsqueeze(-1)).sum() / torch.clamp(valid_elements, min=1.0)


class _ContinuousDeterField(Deter):
    """Use the original Deter displacement as a continuous vector field."""

    def forward(
        self,
        _time: torch.Tensor,
        deter: torch.Tensor,
        args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        stoch, action, duration = args
        unbatched = deter.ndim == 1
        if unbatched:
            deter = deter.unsqueeze(0)
            stoch = stoch.unsqueeze(0)
            action = action.unsqueeze(0)
            duration = duration.unsqueeze(0)
        candidate, update = self._candidate_and_update(stoch, deter, action)
        derivative = duration * update * (candidate - deter)
        return derivative.squeeze(0) if unbatched else derivative


class _TorchODEIntegrator(nn.Module):
    """Integrate deter while passing interval inputs as static ODE arguments."""

    def __init__(
        self,
        ode_func: _ContinuousDeterField,
        method: str,
        rtol: float,
        atol: float,
        max_steps: int,
    ) -> None:
        super().__init__()
        term = torchode.ODETerm(ode_func, with_stats=False, with_args=True)
        object.__setattr__(self, "_term", term)

        if method == "tsit5":
            step_method = torchode.Tsit5()
            step_size_controller = torchode.IntegralController(atol=atol, rtol=rtol)
        elif method == "heun":
            step_method = torchode.Heun()
            step_size_controller = torchode.FixedStepController()
        elif method == "euler":
            step_method = torchode.Euler(term=None)
            step_size_controller = torchode.FixedStepController()
        else:
            raise ValueError(f"Unsupported ode_method: {method!r}.")

        self._solver = torchode.AutoDiffAdjoint(
            step_method,
            step_size_controller,
            max_steps=max_steps,
            backprop_through_step_size_control=False,
        )

    def forward(
        self,
        initial_state: torch.Tensor,
        stoch: torch.Tensor,
        action: torch.Tensor,
        duration: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        initial_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        problem = torchode.InitialValueProblem(
            y0=initial_state,
            t_start=t_start,
            t_end=t_end,
        )
        solution = self._solver.solve(
            problem,
            term=self._term,
            dt0=initial_step,
            args=(stoch, action, duration),
        )
        return solution.ys[:, -1], solution.status


class ContinuousDeter(nn.Module):
    """Continuous version of Deter with linear training and ODE inference."""

    def __init__(
        self,
        deter: int,
        stoch: int,
        act_dim: int,
        hidden: int,
        blocks: int,
        dynlayers: int,
        act: str = "SiLU",
        method: str = "tsit5",
        step_size: float = 0.25,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        max_steps: int = 64,
        compile_solver: bool = True,
    ) -> None:
        super().__init__()
        self.method = str(method)
        self.step_size = float(step_size)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_steps = int(max_steps)
        self.compile_solver = bool(compile_solver)

        self.ode_func = _ContinuousDeterField(
            deter=deter,
            stoch=stoch,
            act_dim=act_dim,
            hidden=hidden,
            blocks=blocks,
            dynlayers=dynlayers,
            act=act,
        )
        self.ode_func.apply(weight_init_)
        self.integrator = _TorchODEIntegrator(
            ode_func=self.ode_func,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )
        solve = torch.compile(self.integrator, fullgraph=False) if self.compile_solver else self.integrator
        object.__setattr__(self, "_solve", solve)

    @staticmethod
    def _prepare_duration(
        delta_t: torch.Tensor | float | None,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if delta_t is None:
            return torch.ones(batch_size, 1, device=reference.device, dtype=torch.float32)

        duration = torch.as_tensor(delta_t, device=reference.device, dtype=torch.float32)
        if duration.ndim == 0:
            duration = duration.expand(batch_size)
        return duration.reshape(batch_size, 1)

    def forward(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor | float | None = None,
        transition_mode: str = "ode",
    ) -> torch.Tensor:
        if transition_mode not in {"linear", "ode"}:
            raise ValueError(f"transition_mode must be 'linear' or 'ode', got {transition_mode!r}.")
        batch_size = action.shape[0]
        flat_stoch = stoch.reshape(batch_size, -1)
        duration = self._prepare_duration(delta_t, batch_size, deter)

        with torch.autocast(device_type=deter.device.type, enabled=False):
            initial_state = deter.float()
            flat_stoch = flat_stoch.float()
            action = action.float()
            duration = duration.float()
            if transition_mode == "linear":
                velocity = self.ode_func(
                    torch.zeros(batch_size, device=deter.device, dtype=torch.float32),
                    initial_state,
                    (flat_stoch, action, torch.ones_like(duration)),
                )
                return initial_state + duration * velocity
            t_start = torch.zeros(batch_size, device=deter.device, dtype=torch.float32)
            t_end = torch.ones(batch_size, device=deter.device, dtype=torch.float32)
            initial_step = torch.full(
                (batch_size,),
                self.step_size,
                device=deter.device,
                dtype=torch.float32,
            )
            final_state, status = self._solve(
                initial_state,
                flat_stoch,
                action,
                duration,
                t_start,
                t_end,
                initial_step,
            )
        torch._assert_async((status == 0).all(), "torchode integration failed.")
        return final_state

    def vector_field(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the base-duration vector field without integrating it."""
        batch_size = action.shape[0]
        flat_stoch = stoch.reshape(batch_size, -1)
        with torch.autocast(device_type=deter.device.type, enabled=False):
            return self.ode_func(
                torch.zeros(batch_size, device=deter.device, dtype=torch.float32),
                deter.float(),
                (
                    flat_stoch.float(),
                    action.float(),
                    torch.ones(batch_size, 1, device=deter.device, dtype=torch.float32),
                ),
            )


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        transition_type = str(getattr(config, "transition_type", "discrete"))
        transition_kwargs = {
            "deter": self._deter,
            "stoch": self.flat_stoch,
            "act_dim": act_dim,
            "hidden": self._hidden,
            "blocks": self._blocks,
            "dynlayers": self._dyn_layers,
            "act": config.act,
        }
        if transition_type == "discrete":
            self._deter_net = Deter(**transition_kwargs)
        elif transition_type == "continuous":
            self._deter_net = ContinuousDeter(
                **transition_kwargs,
                method=str(config.ode_method),
                step_size=float(config.ode_step_size),
                rtol=float(config.ode_rtol),
                atol=float(config.ode_atol),
                max_steps=int(getattr(config, "ode_max_steps", 64)),
                compile_solver=bool(getattr(config, "ode_compile", False)),
            )
        else:
            raise ValueError(f"transition_type must be 'discrete' or 'continuous', got {transition_type!r}.")

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset, delta_t=None, transition_mode="ode"):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            step_delta_t = None if delta_t is None else delta_t[:, i]
            stoch, deter, logit = self.obs_step(
                stoch,
                deter,
                action[:, i],
                embed[:, i],
                reset[:, i],
                delta_t=step_delta_t,
                transition_mode=transition_mode,
            )
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset, delta_t=None, transition_mode="ode"):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,)
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        deter = self._transition(stoch, deter, prev_action, delta_t, transition_mode)
        # (B, D + E)
        x = torch.cat([deter, embed], dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action, delta_t=None, transition_mode="ode"):
        """Single prior step (no observation)."""

        # (B, D)
        deter = self._transition(stoch, deter, prev_action, delta_t, transition_mode)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions, delta_t=None, transition_mode="ode"):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            step_delta_t = None if delta_t is None else delta_t[:, i]
            stoch, deter = self.img_step(
                stoch,
                deter,
                actions[:, i],
                delta_t=step_delta_t,
                transition_mode=transition_mode,
            )
            stochs.append(stoch)
            deters.append(deter)
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def _transition(self, stoch, deter, action, delta_t, transition_mode):
        if isinstance(self._deter_net, ContinuousDeter):
            return self._deter_net(
                stoch,
                deter,
                action,
                delta_t=delta_t,
                transition_mode=transition_mode,
            )
        return self._deter_net(stoch, deter, action)

    def deter_velocity(self, stoch, deter, action):
        """Evaluate the continuous deterministic vector field."""
        if not isinstance(self._deter_net, ContinuousDeter):
            raise RuntimeError("deter_velocity is only available for the continuous transition.")
        return self._deter_net.vector_field(stoch, deter, action)

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
