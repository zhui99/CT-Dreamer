import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torchode

import dreamer
import rssm


def _small_deter(**overrides):
    kwargs = {
        "deter": 8,
        "stoch": 6,
        "act_dim": 3,
        "hidden": 4,
        "blocks": 2,
        "dynlayers": 1,
        "act": "SiLU",
        "method": "tsit5",
        "step_size": 0.25,
        "rtol": 1e-4,
        "atol": 1e-5,
        "max_steps": 64,
        "compile_solver": False,
    }
    return rssm.ContinuousDeter(**(kwargs | overrides))


def _inputs(batch_size=2):
    generator = torch.Generator().manual_seed(7)
    stoch = torch.randn(batch_size, 2, 3, generator=generator)
    deter = torch.randn(batch_size, 8, generator=generator)
    action = torch.randn(batch_size, 3, generator=generator)
    return stoch, deter, action


def _rssm_config(transition_type="continuous"):
    return SimpleNamespace(
        stoch=2,
        deter=8,
        hidden=4,
        discrete=3,
        unimix_ratio=0.01,
        initial="zeros",
        device="cpu",
        obs_layers=1,
        img_layers=1,
        dyn_layers=1,
        blocks=2,
        act="SiLU",
        transition_type=transition_type,
        ode_method="tsit5",
        ode_step_size=0.25,
        ode_rtol=1e-4,
        ode_atol=1e-5,
        ode_max_steps=64,
        ode_compile=False,
    )


def test_legacy_gated_leaky_api_is_removed():
    assert not hasattr(rssm, "GatedLeakyDeter")
    assert not hasattr(rssm, "GatedLeakyODEFunc")
    assert not hasattr(rssm, "gate_to_rate")


def test_internal_validation_helpers_are_removed():
    assert not any(
        (
            hasattr(rssm, "_validate_duration_values"),
            hasattr(rssm, "_check_ode_status"),
            hasattr(rssm, "_EagerIntegrator"),
            hasattr(rssm.ContinuousDeter, "_SUPPORTED_METHODS"),
        )
    )


def test_continuous_deter_uses_autodiff_tsit5():
    model = _small_deter()
    integrator = model.integrator

    assert integrator._term.with_args
    assert isinstance(integrator._solver, torchode.AutoDiffAdjoint)
    assert isinstance(integrator._solver.step_method, torchode.Tsit5)
    assert isinstance(integrator._solver.step_size_controller, torchode.IntegralController)
    assert integrator._solver.max_steps == 64
    assert not integrator._solver.backprop_through_step_size_control


def test_ode_function_only_integrates_deter_and_receives_static_args():
    model = _small_deter()
    stoch, deter, action = _inputs()
    duration = torch.ones(deter.shape[0], 1)
    static_args = (stoch.flatten(1), action, duration)

    derivative = model.ode_func(
        torch.zeros(deter.shape[0]),
        deter,
        static_args,
    )

    assert derivative.shape == deter.shape


def test_linear_transition_matches_forward_euler_without_calling_solver():
    model = _small_deter()
    stoch, deter, action = _inputs()
    duration = torch.tensor([0.5, 1.5])

    velocity = model.vector_field(stoch, deter, action)
    with mock.patch.object(model.integrator, "forward", side_effect=AssertionError("solver called")):
        output = model(stoch, deter, action, delta_t=duration, transition_mode="linear")

    expected = deter + duration.unsqueeze(-1) * velocity
    torch.testing.assert_close(output, expected)


def test_unit_linear_transition_exactly_matches_original_deter():
    original = rssm.Deter(deter=8, stoch=6, act_dim=3, hidden=4, blocks=2, dynlayers=1)
    original.apply(rssm.weight_init_)
    continuous = _small_deter()
    continuous.ode_func.load_state_dict(original.state_dict())
    stoch, deter, action = _inputs()

    expected = original(stoch, deter, action)
    actual = continuous(stoch, deter, action, delta_t=1.0, transition_mode="linear")

    torch.testing.assert_close(actual, expected)


def test_linear_transition_supports_fullgraph_compile_with_duration():
    model = _small_deter()
    stoch, deter, action = _inputs()
    duration = torch.tensor([0.5, 1.5])
    compiled_model = torch.compile(model, backend="eager", fullgraph=True)

    output = compiled_model(stoch, deter, action, delta_t=duration, transition_mode="linear")

    assert output.shape == deter.shape


def test_ode_transition_remains_the_default():
    model = _small_deter(method="euler", step_size=1.0)
    stoch, deter, action = _inputs()

    with mock.patch.object(model.integrator, "forward", wraps=model.integrator.forward) as solve:
        output = model(stoch, deter, action)

    assert output.shape == deter.shape
    assert solve.call_count == 1


def test_unknown_transition_mode_is_rejected():
    model = _small_deter()
    stoch, deter, action = _inputs()

    with pytest.raises(ValueError, match="transition_mode"):
        model(stoch, deter, action, transition_mode="midpoint")


def test_euler_solver_supports_forward_and_backward():
    model = _small_deter(method="euler", step_size=1.0)
    stoch, deter, action = _inputs()

    output = model(stoch, deter, action)
    output.square().mean().backward()

    assert output.shape == deter.shape
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_compile_toggle_preserves_state_dict_and_deepcopy_ownership():
    eager = _small_deter(compile_solver=False)
    compiled = _small_deter(compile_solver=True)
    cloned = copy.deepcopy(compiled)

    assert set(eager.state_dict()) == set(compiled.state_dict())
    assert set(compiled.state_dict()) == set(cloned.state_dict())
    assert next(compiled.parameters()).data_ptr() != next(cloned.parameters()).data_ptr()
    assert eager.integrator._term.f is eager.ode_func
    assert compiled.integrator._term.f is compiled.ode_func
    assert cloned.integrator._term.f is cloned.ode_func
    assert compiled._solve._orig_mod is compiled.integrator
    assert cloned._solve._orig_mod is cloned.integrator


def test_zero_duration_is_identity():
    model = _small_deter()
    stoch, deter, action = _inputs()
    delta_t = torch.zeros(deter.shape[0])

    output = model(stoch, deter, action, delta_t=delta_t)

    torch.testing.assert_close(output, deter)


@pytest.mark.parametrize("delta_t", [0.5, torch.tensor([0.5, 1.5]), torch.ones(2, 1)])
def test_supported_duration_shapes(delta_t):
    model = _small_deter()
    stoch, deter, action = _inputs()

    output = model(stoch, deter, action, delta_t=delta_t)

    assert output.shape == deter.shape


def test_adjoint_backward_produces_finite_gradients():
    model = _small_deter()
    stoch, deter, action = _inputs()
    stoch.requires_grad_(True)
    deter.requires_grad_(True)
    action.requires_grad_(True)
    duration = torch.tensor([0.5, 1.5], requires_grad=True)

    model(stoch, deter, action, delta_t=duration).square().mean().backward()

    parameter_grads = [parameter.grad for parameter in model.parameters()]
    assert parameter_grads
    assert all(gradient is not None for gradient in parameter_grads)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_grads)
    assert torch.isfinite(stoch.grad).all()
    assert torch.isfinite(deter.grad).all()
    assert torch.isfinite(action.grad).all()
    assert torch.isfinite(duration.grad).all()
    assert duration.grad.abs().max() > 0


def test_rssm_selects_continuous_transition():
    model = rssm.RSSM(_rssm_config(), embed_size=5, act_dim=3)

    assert isinstance(model._deter_net, rssm.ContinuousDeter)


def test_discrete_deter_remains_available():
    model = rssm.RSSM(_rssm_config("discrete"), embed_size=5, act_dim=3)
    _, deter = model.initial(batch_size=2)
    stoch = torch.nn.functional.one_hot(torch.zeros(2, 2, dtype=torch.long), num_classes=3).float()
    action = torch.zeros(2, 3)

    output = model._deter_net(stoch, deter, action)

    assert isinstance(model._deter_net, rssm.Deter)
    assert output.shape == deter.shape


def test_rssm_observe_and_imagine_with_continuous_deter():
    model = rssm.RSSM(_rssm_config(), embed_size=5, act_dim=3)
    batch_size, sequence_length = 2, 3
    initial = model.initial(batch_size)
    embed = torch.randn(batch_size, sequence_length, 5)
    actions = torch.randn(batch_size, sequence_length, 3)
    reset = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    reset[:, 0] = True

    post_stoch, post_deter, post_logits = model.observe(embed, actions, initial, reset)
    _, prior_logits = model.prior(post_deter)
    dyn_loss, rep_loss = model.kl_loss(post_logits, prior_logits, free=0.1)
    imag_stoch, imag_deter = model.imagine_with_action(post_stoch[:, -1], post_deter[:, -1], actions)
    features = model.get_feat(imag_stoch, imag_deter)

    assert post_stoch.shape == (batch_size, sequence_length, 2, 3)
    assert post_deter.shape == (batch_size, sequence_length, 8)
    assert dyn_loss.shape == (batch_size, sequence_length)
    assert rep_loss.shape == (batch_size, sequence_length)
    assert features.shape == (batch_size, sequence_length, model.feat_size)


def test_rssm_linear_observe_skips_solver_but_imagination_uses_it():
    model = rssm.RSSM(_rssm_config(), embed_size=5, act_dim=3)
    batch_size, sequence_length = 2, 3
    initial = model.initial(batch_size)
    embed = torch.randn(batch_size, sequence_length, 5)
    actions = torch.randn(batch_size, sequence_length, 3)
    reset = torch.zeros(batch_size, sequence_length, dtype=torch.bool)

    with mock.patch.object(
        model._deter_net.integrator,
        "forward",
        side_effect=AssertionError("solver called during linear observe"),
    ):
        post_stoch, post_deter, _ = model.observe(
            embed,
            actions,
            initial,
            reset,
            transition_mode="linear",
        )

    with mock.patch.object(
        model._deter_net.integrator,
        "forward",
        wraps=model._deter_net.integrator.forward,
    ) as solve:
        model.imagine_with_action(post_stoch[:, -1], post_deter[:, -1], actions)

    assert solve.call_count == sequence_length


def test_linear_interval_velocity_loss_uses_duration_and_detaches_targets():
    predicted = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]], requires_grad=True)
    embeddings = torch.tensor(
        [[[0.0, 0.0], [2.0, 4.0], [5.0, 10.0]]],
        requires_grad=True,
    )
    duration = torch.tensor([[1.0, 2.0, 1.0]])
    reset = torch.zeros(1, 3, dtype=torch.bool)

    loss = rssm.linear_interval_velocity_loss(predicted, embeddings, duration, reset)
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert predicted.grad is not None
    assert embeddings.grad is None


def test_linear_interval_velocity_loss_computes_in_float32_for_half_inputs():
    predicted = torch.zeros(1, 1, 1, dtype=torch.float16, requires_grad=True)
    embeddings = torch.tensor([[[0.0], [1.0e-4]]], dtype=torch.float16)
    duration = torch.tensor([[1.0, 1.0e-4]], dtype=torch.float16)

    loss = rssm.linear_interval_velocity_loss(predicted, embeddings, duration)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_linear_interval_velocity_loss_supports_fullgraph_compile():
    predicted = torch.zeros(1, 1, 1)
    embeddings = torch.tensor([[[0.0], [1.0]]])
    duration = torch.tensor([[1.0, 1.0]])
    compiled_loss = torch.compile(
        rssm.linear_interval_velocity_loss,
        backend="eager",
        fullgraph=True,
    )

    loss = compiled_loss(predicted, embeddings, duration)

    torch.testing.assert_close(loss, torch.ones_like(loss))


def test_linear_interval_velocity_loss_ignores_reset_and_zero_duration():
    predicted = torch.tensor([[[100.0], [100.0]]], requires_grad=True)
    embeddings = torch.tensor([[[0.0], [1.0], [2.0]]])
    duration = torch.tensor([[1.0, 1.0, 0.0]])
    reset = torch.tensor([[False, True, False]])

    loss = rssm.linear_interval_velocity_loss(predicted, embeddings, duration, reset)
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))


def test_dreamer_velocity_loss_aligns_next_action_and_projected_deter_rate():
    class FakeRSSM:
        flat_stoch = 6

        @staticmethod
        def deter_velocity(stoch, deter, action):
            del stoch, deter
            return torch.cat([action, 2.0 * action], dim=-1)

    agent = SimpleNamespace(rssm=FakeRSSM())
    agent.prj = torch.nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        agent.prj.weight.zero_()
        agent.prj.weight[:, -2:] = torch.eye(2)

    post_stoch = torch.zeros(1, 3, 2, 3)
    post_deter = torch.zeros(1, 3, 2)
    action = torch.tensor([[[99.0], [1.0], [3.0]]])
    embed = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [4.0, 8.0]]], requires_grad=True)
    reset = torch.zeros(1, 3, dtype=torch.bool)

    loss = dreamer.Dreamer._linear_velocity_loss(
        agent,
        post_stoch,
        post_deter,
        action,
        embed,
        delta_t=None,
        reset=reset,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert agent.prj.weight.grad is not None
    assert embed.grad is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "dopri5"}, "ode_method"),
        ({"blocks": 3}, "divisible"),
    ],
)
def test_invalid_algorithm_configuration_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _small_deter(**overrides)


def test_unknown_transition_type_is_rejected():
    with pytest.raises(ValueError, match="transition_type"):
        rssm.RSSM(_rssm_config("unknown"), embed_size=5, act_dim=3)
