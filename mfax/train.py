"""
Thin callable wrappers around HSM algorithm training pipelines.
Imports make_train_step from the original algorithm files; does not duplicate logic.
"""

from dataclasses import dataclass, fields
from typing import Any, List
import time

import gymnax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from mfax.algos.hsm.exploitability import make_exploitability
from mfax.envs import make_env


@dataclass
class TrainResult:
    algo: str
    iterations: List[int]
    train_times: List[float]
    exploitabilities: List[float]
    policy_returns: List[float]
    final_eval_results: Any
    final_params: Any


def train(algo: str, max_time: float = 300.0, **overrides) -> TrainResult:
    """Train SPG or RSPG and return metrics + final evaluation results."""

    # --- import algo-specific modules ---
    if algo == "spg":
        from mfax.algos.hsm.algos.spg import args as ArgsClass, make_train_step
        from mfax.algos.hsm.utils.mf_policy_wrappers import (
            MeanFieldPolicy,
            MeanFieldContinuousPolicy,
        )
        from mfax.algos.hsm.utils.make_act import MFActorWrapper

        is_recurrent = False
    elif algo == "rspg":
        from mfax.algos.hsm.algos.rspg import args as ArgsClass, make_train_step
        from mfax.algos.hsm.utils.mf_policy_wrappers import (
            RecurrentMeanFieldPolicy,
            RecurrentMeanFieldContinuousPolicy,
        )
        from mfax.algos.hsm.utils.make_act import MFRecurrentActorWrapper

        is_recurrent = True
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'spg' or 'rspg'.")

    # --- create args with overrides ---
    args = ArgsClass()
    valid_fields = {f.name for f in fields(ArgsClass)}
    for k, v in overrides.items():
        if k in valid_fields:
            object.__setattr__(args, k, v)
    # force evaluation on, logging/saving off for notebook use
    args.evaluate = True
    args.log = False
    args.save = False
    args.debug = False

    if args.state_type == "indices" and args.normalize_states:
        args.normalize_states = False

    # --- make environment ---
    env = make_env("pushforward/" + args.task, common_noise=args.common_noise)

    # --- make policy network ---
    if is_recurrent:
        encoder_kwargs = dict(
            hidden_size=128, embed_size=128, activation=args.activation
        )
        policy_kwargs = dict(
            hidden_layer_sizes=(128, 128, 128),
            activation=args.activation,
            state_type=args.state_type,
            num_states=env.n_states,
        )
        if isinstance(env.action_space, gymnax.environments.spaces.Discrete):
            policy_kwargs["n_actions"] = env.n_actions
            mf_policy_net = RecurrentMeanFieldPolicy(
                state_type=args.state_type,
                num_states=env.n_states,
                encoder_kwargs=encoder_kwargs,
                policy_kwargs=policy_kwargs,
            )
        else:
            policy_kwargs["action_dim"] = env.action_space.shape[-1]
            policy_kwargs["action_range"] = (
                env.action_space.low,
                env.action_space.high,
            )
            mf_policy_net = RecurrentMeanFieldContinuousPolicy(
                state_type=args.state_type,
                num_states=env.n_states,
                actions=env.params.discrete_actions,
                encoder_kwargs=encoder_kwargs,
                policy_kwargs=policy_kwargs,
            )
    else:
        policy_kwargs = dict(
            hidden_layer_sizes=(128, 128, 128),
            activation=args.activation,
            state_type=args.state_type,
            num_states=env.n_states,
        )
        if isinstance(env.action_space, gymnax.environments.spaces.Discrete):
            policy_kwargs["n_actions"] = env.n_actions
            mf_policy_net = MeanFieldPolicy(
                state_type=args.state_type,
                num_states=env.n_states,
                policy_kwargs=policy_kwargs,
            )
        else:
            policy_kwargs["action_dim"] = env.action_space.shape[-1]
            policy_kwargs["action_range"] = (
                env.action_space.low,
                env.action_space.high,
            )
            mf_policy_net = MeanFieldContinuousPolicy(
                state_type=args.state_type,
                num_states=env.n_states,
                actions=env.params.discrete_actions,
                policy_kwargs=policy_kwargs,
            )

    # --- individual states ---
    if args.state_type == "indices":
        individual_states = jnp.arange(env.n_states)
    else:
        individual_states = env.params.states

    # --- init params ---
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_actor_params, rng_action = jax.random.split(rng, 3)
    init_obs = jnp.ones((args.num_envs, env.obs_dim), dtype=jnp.float32)

    if is_recurrent:
        init_hidden = mf_policy_net.init_hidden(
            args.num_envs, mf_policy_net.aggregate_encoder.hidden_size
        )
        init_done = jnp.zeros((args.num_envs,), dtype=jnp.bool_)
        actor_params = mf_policy_net.init(
            rng_actor_params,
            env.normalize_individual_s(individual_states, args.normalize_states),
            env.normalize_obs(init_obs, args.normalize_obs),
            init_hidden,
            init_done,
            rng_action,
        )
    else:
        actor_params = mf_policy_net.init(
            rng_actor_params,
            env.normalize_individual_s(individual_states, args.normalize_states),
            env.normalize_obs(init_obs, args.normalize_obs),
            rng_action,
        )

    # --- optimizer and train state ---
    if args.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(
                learning_rate=optax.linear_schedule(
                    args.lr, args.lr * 0.1, args.num_iterations
                ),
                eps=1e-8,
            ),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-8),
        )
    actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)

    # --- JIT train step ---
    train_step = jax.jit(make_train_step(args, env, mf_policy_net, individual_states))

    # --- exploitability evaluator ---
    if is_recurrent:
        mf_agent_wrapper = MFRecurrentActorWrapper(
            mf_policy_net,
            actor_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
    else:
        mf_agent_wrapper = MFActorWrapper(
            mf_policy_net,
            actor_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
    exploitability_fn = make_exploitability(
        env=env,
        agent=mf_agent_wrapper,
        state_type=args.state_type,
        gamma=args.discount_factor,
        num_envs=args.num_envs,
        max_steps_in_episode=env.params.max_steps_in_episode,
    )

    # --- JIT warmup: trigger compilation before timing ---
    print(f"[{algo.upper()}] Compiling (JIT warmup)...")
    rng, warmup_rng, warmup_eval_rng = jax.random.split(rng, 3)
    warmup_state = (actor_ts, warmup_rng)
    warmup_state, _ = jax.lax.scan(train_step, warmup_state, None, 1)
    jax.block_until_ready(warmup_state)
    _ = exploitability_fn(warmup_eval_rng, actor_ts.params)
    print(f"[{algo.upper()}] Compilation done.")

    # --- training loop ---
    result = TrainResult(
        algo=algo,
        iterations=[],
        train_times=[],
        exploitabilities=[],
        policy_returns=[],
        final_eval_results=None,
        final_params=None,
    )

    elapsed_time = 0.0

    # initial eval
    rng, eval_rng = jax.random.split(rng)
    mf_eval_results = exploitability_fn(eval_rng, actor_ts.params)
    result.iterations.append(0)
    result.train_times.append(0.0)
    result.exploitabilities.append(float(mf_eval_results.exploitability.exploitability))
    result.policy_returns.append(
        float(mf_eval_results.exploitability.mean_policy_return)
    )
    print(
        f"[{algo.upper()}] Iter 0, Time 0.0s, "
        f"Exploitability: {mf_eval_results.exploitability.exploitability:.6f}"
    )

    for iteration_idx in range(0, args.num_iterations, args.eval_frequency):
        iteration_idx += args.eval_frequency

        t0 = time.perf_counter()
        runner_state = (actor_ts, rng)
        runner_state, _ = jax.lax.scan(
            train_step, runner_state, None, args.eval_frequency
        )
        (actor_ts, rng) = runner_state
        jax.block_until_ready(runner_state)
        elapsed_time += time.perf_counter() - t0

        # eval
        rng, eval_rng = jax.random.split(rng)
        mf_eval_results = exploitability_fn(eval_rng, actor_ts.params)

        result.iterations.append(iteration_idx)
        result.train_times.append(elapsed_time)
        result.exploitabilities.append(
            float(mf_eval_results.exploitability.exploitability)
        )
        result.policy_returns.append(
            float(mf_eval_results.exploitability.mean_policy_return)
        )
        print(
            f"[{algo.upper()}] Iter {iteration_idx}, Time {elapsed_time:.1f}s, "
            f"Exploitability: {mf_eval_results.exploitability.exploitability:.6f}"
        )

        if elapsed_time >= max_time:
            print(f"[{algo.upper()}] Time limit ({max_time}s) reached, stopping.")
            break

    result.final_eval_results = mf_eval_results
    result.final_params = actor_ts.params
    return result
