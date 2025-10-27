'''
Gradient checking utility for verifying analytical gradients against numerical approximations.
Useful for debugging gradient computation issues in the IHDP agent.
'''
import torch
import numpy as np
import sys
import os

from ihdp import IHDPAgent


class GradientChecker:
    """Utility class for performing gradient checking on IHDP agent networks."""

    @staticmethod
    def check_agent_gradients(agent: 'IHDPAgent', obs: np.ndarray, next_obs: np.ndarray,
                             reward: float, epsilon: float = 1e-5,
                             tolerance: float = 1e-3, check_rls_actor: bool = True) -> dict:
        """
        Perform gradient checking for both actor and critic networks of an IHDP agent.

        Compares analytical gradients computed via backpropagation with numerical gradients
        estimated using finite differences.

        Args:
            agent: IHDPAgent instance
            obs: Current observation
            next_obs: Next observation
            reward: Reward received
            epsilon: Perturbation size for finite differences (default 1e-5)
            tolerance: Maximum relative difference between analytical and numerical gradients (default 1e-3)
            check_rls_actor: If True, check RLS-based actor gradients (default True)

        Returns:
            Dictionary containing gradient check results for both networks
        """
        if check_rls_actor:
            # Check actor using RLS Jacobian method
            actor_results = GradientChecker.check_actor_gradients_with_rls(
                agent.actor, agent.critic, agent.model, obs, epsilon, tolerance
            )
        else:
            # Standard backprop check (won't work for IHDP actor)
            actor_results = GradientChecker.check_actor_gradients(
                agent.actor, agent.critic, obs, epsilon, tolerance
            )

        critic_results = GradientChecker.check_critic_gradients(
            agent.critic, obs, next_obs, reward, agent.gamma, epsilon, tolerance
        )

        all_passed = actor_results['passed'] and critic_results['passed']

        return {
            'actor': actor_results,
            'critic': critic_results,
            'all_passed': all_passed,
        }

    @staticmethod
    def check_actor_gradients_with_rls(actor, critic, model, obs: np.ndarray,
                                       epsilon: float = 1e-5, tolerance: float = 1e-3) -> dict:
        """
        Perform gradient checking for the actor network using RLS-based manual gradients.

        For IHDP, the actor uses manual gradient computation:
        dL/du = -dV/dx * G (where G is the control sensitivity from RLS model)

        This checks that the manual gradient application is correct by verifying:
        dL/dtheta = dL/du * du/dtheta is computed correctly.

        Args:
            actor: Actor network module
            critic: Critic network module
            model: RLS model for getting Jacobians
            obs: Current observation
            epsilon: Perturbation size for finite differences (default 1e-5)
            tolerance: Maximum relative difference between analytical and numerical gradients (default 1e-3)

        Returns:
            Dictionary containing gradient check results
        """
        obs_torch = torch.FloatTensor(obs).unsqueeze(0)
        obs_torch.requires_grad_(True)

        # Compute dV/dx through critic by backpropagating through critic
        V_pred = critic(obs_torch)
        V_pred.backward()
        dVdx = obs_torch.grad.detach().numpy().squeeze()

        # Get RLS model Jacobian G
        _, G_t = model.get_jacobians()

        # Manual gradient computation: dL/du = -dV/dx * G
        manual_grad_u = -dVdx @ G_t  # shape: (act_dim,)

        # Compute numerical gradients by perturbing actor parameters
        errors = []

        for param in actor.parameters():
            original_data = param.data.clone()
            param_size = param.data.numel()

            # Check first 50 params for efficiency
            for flat_idx in range(min(param_size, 50)):
                multi_idx = np.unravel_index(flat_idx, param.shape)

                # Compute action after perturbing parameter by +epsilon
                param.data = original_data.clone()
                param.data[multi_idx] += epsilon
                with torch.no_grad():
                    action_plus = actor(obs_torch.detach()).squeeze().numpy()
                    V_plus = critic(obs_torch.detach()).item()

                # Compute action after perturbing parameter by -epsilon
                param.data = original_data.clone()
                param.data[multi_idx] -= epsilon
                with torch.no_grad():
                    action_minus = actor(obs_torch.detach()).squeeze().numpy()
                    V_minus = critic(obs_torch.detach()).item()

                # Numerical gradient via finite differences: dL/dtheta = dV/dtheta
                numerical_grad_param = (V_plus - V_minus) / (2 * epsilon)

                # Derivative of action w.r.t. parameter
                du_dtheta = (action_plus - action_minus) / (2 * epsilon)

                # Expected gradient: dL/dtheta = dL/du * du/dtheta
                expected_grad_param = np.dot(manual_grad_u, du_dtheta)

                # Relative error
                denominator = max(abs(expected_grad_param), abs(numerical_grad_param), 1e-8)
                relative_error = abs(expected_grad_param - numerical_grad_param) / denominator
                errors.append(relative_error)

                param.data = original_data.clone()

        obs_torch.requires_grad_(False)

        passed = all(e < tolerance for e in errors) if errors else False

        return {
            'errors': errors,
            'passed': passed,
            'max_error': max(errors) if errors else 0.0,
            'avg_error': np.mean(errors) if errors else 0.0,
            'num_checks': len(errors),
        }

    @staticmethod
    def check_actor_gradients(actor, critic, obs: np.ndarray, epsilon: float = 1e-5,
                              tolerance: float = 1e-3) -> dict:
        """
        Perform gradient checking for the actor network.

        Compares analytical gradients computed via backpropagation with numerical gradients
        estimated using finite differences.

        Args:
            actor: Actor network module
            critic: Critic network module
            obs: Current observation
            epsilon: Perturbation size for finite differences (default 1e-5)
            tolerance: Maximum relative difference between analytical and numerical gradients (default 1e-3)

        Returns:
            Dictionary containing:
                - 'errors': List of relative errors for each actor parameter
                - 'passed': Boolean indicating if all gradients passed tolerance check
                - 'max_error': Maximum relative error
                - 'avg_error': Average relative error
        """
        obs_torch = torch.FloatTensor(obs).unsqueeze(0)
        obs_torch.requires_grad_(True)

        # Compute analytical gradients for actor
        actor_action = actor(obs_torch)
        actor_action.retain_grad()

        # Forward pass through critic to get loss
        V_pred = critic(obs_torch)
        actor_loss = -V_pred.mean()

        actor.zero_grad()
        actor_loss.backward()

        # Store analytical gradients
        analytical_grads = []
        for param in actor.parameters():
            if param.grad is not None:
                analytical_grads.append(param.grad.detach().clone())

        # Compute numerical gradients
        errors = []
        grad_param_idx = 0
        for param_idx, param in enumerate(actor.parameters()):
            original_data = param.data.clone()
            param_size = param.data.numel()

            # Skip if no gradient for this parameter
            if grad_param_idx >= len(analytical_grads):
                continue

            for flat_idx in range(param_size):
                # Reshape flat index to parameter shape
                multi_idx = np.unravel_index(flat_idx, param.shape)

                # Compute f(x + epsilon)
                param.data = original_data.clone()
                param.data[multi_idx] += epsilon
                with torch.no_grad():
                    V_plus = critic(obs_torch.detach())
                    loss_plus = -V_plus.mean().item()

                # Compute f(x - epsilon)
                param.data = original_data.clone()
                param.data[multi_idx] -= epsilon
                with torch.no_grad():
                    V_minus = critic(obs_torch.detach())
                    loss_minus = -V_minus.mean().item()

                # Numerical gradient
                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                analytical_grad = analytical_grads[grad_param_idx].flatten()[flat_idx].item()

                # Compute relative error
                denominator = max(abs(analytical_grad), abs(numerical_grad), 1.0)
                relative_error = abs(analytical_grad - numerical_grad) / denominator
                errors.append(relative_error)

                param.data = original_data.clone()

            grad_param_idx += 1

        obs_torch.requires_grad_(False)

        passed = all(e < tolerance for e in errors) if errors else False

        return {
            'errors': errors,
            'passed': passed,
            'max_error': max(errors) if errors else 0.0,
            'avg_error': np.mean(errors) if errors else 0.0,
            'num_checks': len(errors),
        }

    @staticmethod
    def check_critic_gradients(critic, obs: np.ndarray, next_obs: np.ndarray, reward: float,
                               gamma: float = 0.99, epsilon: float = 1e-5,
                               tolerance: float = 1e-3, target_value: torch.Tensor = None) -> dict:
        """
        Perform gradient checking for the critic network.

        Args:
            critic: Critic network module
            obs: Current observation
            next_obs: Next observation
            reward: Reward received
            gamma: Discount factor (default 0.99)
            epsilon: Perturbation size for finite differences (default 1e-5)
            tolerance: Maximum relative difference between analytical and numerical gradients (default 1e-3)

        Returns:
            Dictionary containing:
                - 'errors': List of relative errors for each critic parameter
                - 'passed': Boolean indicating if all gradients passed tolerance check
                - 'max_error': Maximum relative error
                - 'avg_error': Average relative error
        """
        obs_torch = torch.FloatTensor(obs).unsqueeze(0)
        next_obs_torch = torch.FloatTensor(next_obs).unsqueeze(0)
        reward_torch = torch.FloatTensor([reward]).unsqueeze(0)

        obs_torch.requires_grad_(True)

        # Compute analytical gradients for critic
        with torch.no_grad():
            target_value = reward_torch + gamma * critic(next_obs_torch)

        value_pred = critic(obs_torch)
        critic_loss = 0.5 * (target_value - value_pred).pow(2).mean()

        critic.zero_grad()
        critic_loss.backward()

        # Store analytical gradients
        analytical_grads = []
        for param in critic.parameters():
            if param.grad is not None:
                analytical_grads.append(param.grad.detach().clone())

        # Compute numerical gradients
        errors = []
        grad_param_idx = 0
        for param in critic.parameters():
            original_data = param.data.clone()
            param_size = param.data.numel()

            # Skip if no gradient for this parameter
            if grad_param_idx >= len(analytical_grads):
                continue

            # Check only first 10 params per layer for efficiency
            for flat_idx in range(min(param_size, 10)):
                # Reshape flat index to parameter shape
                multi_idx = np.unravel_index(flat_idx, param.shape)

                # Compute f(x + epsilon)
                param.data = original_data.clone()
                param.data[multi_idx] += epsilon
                with torch.no_grad():
                    V_plus = critic(obs_torch.detach())
                    loss_plus = 0.5 * (target_value.detach() - V_plus).pow(2).mean().item()

                # Compute f(x - epsilon)
                param.data = original_data.clone()
                param.data[multi_idx] -= epsilon
                with torch.no_grad():
                    V_minus = critic(obs_torch.detach())
                    loss_minus = 0.5 * (target_value.detach() - V_minus).pow(2).mean().item()

                # Numerical gradient
                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                analytical_grad = analytical_grads[grad_param_idx].flatten()[flat_idx].item()

                # Compute relative error
                denominator = max(abs(analytical_grad), abs(numerical_grad), 1.0)
                relative_error = abs(analytical_grad - numerical_grad) / denominator
                errors.append(relative_error)

                param.data = original_data.clone()

            grad_param_idx += 1

        obs_torch.requires_grad_(False)

        passed = all(e < tolerance for e in errors) if errors else False

        return {
            'errors': errors,
            'passed': passed,
            'max_error': max(errors) if errors else 0.0,
            'avg_error': np.mean(errors) if errors else 0.0,
            'num_checks': len(errors),
        }

    @staticmethod
    def print_gradient_check_results(results: dict):
        """
        Print gradient checking results in a readable format.

        Args:
            results: Results from check_agent_gradients() or can also accept
                    separate actor_results and critic_results dicts
        """
        # Handle both combined and separate result formats
        if 'actor' in results and 'critic' in results:
            actor_results = results['actor']
            critic_results = results['critic']
            all_passed = results.get('all_passed', False)
        else:
            # Legacy format for separate calls
            print("\n" + "="*60)
            print("GRADIENT CHECKING RESULTS")
            print("="*60)
            print("\nNETWORK:")
            print(f"  Passed: {'PASS' if results.get('passed') else 'FAIL'}")
            print(f"  Max Error: {results.get('max_error', 0):.2e}")
            print(f"  Avg Error: {results.get('avg_error', 0):.2e}")
            print(f"  Num Checks: {results.get('num_checks', 0)}")
            return

        print("\n" + "="*60)
        print("GRADIENT CHECKING RESULTS")
        print("="*60)

        print("\nACTOR NETWORK:")
        print(f"  Passed: {'PASS' if actor_results['passed'] else 'FAIL'}")
        max_error = np.asarray(actor_results['max_error']).item() if np.asarray(actor_results['max_error']).size == 1 else float(np.max(actor_results['max_error']))
        avg_error = np.asarray(actor_results['avg_error']).item() if np.asarray(actor_results['avg_error']).size == 1 else float(np.mean(actor_results['avg_error']))
        print(f"  Max Error: {max_error:.2e}")
        print(f"  Avg Error: {avg_error:.2e}")
        print(f"  Num Checks: {actor_results['num_checks']}")

        print("\nCRITIC NETWORK:")
        print(f"  Passed: {'PASS' if critic_results['passed'] else 'FAIL'}")
        max_error_c = np.asarray(critic_results['max_error']).item() if np.asarray(critic_results['max_error']).size == 1 else float(np.max(critic_results['max_error']))
        avg_error_c = np.asarray(critic_results['avg_error']).item() if np.asarray(critic_results['avg_error']).size == 1 else float(np.mean(critic_results['avg_error']))
        print(f"  Max Error: {max_error_c:.2e}")
        print(f"  Avg Error: {avg_error_c:.2e}")
        print(f"  Num Checks: {critic_results['num_checks']}")

        print("\n" + "="*60)
        if all_passed:
            print("SUCCESS: All gradient checks PASSED!")
        else:
            print("FAILURE: Some gradient checks FAILED. Check implementation.")
        print("="*60 + "\n")


def main():
    """Run gradient checking on IHDP agent."""
    # Add parent directories to path for imports
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../envs/pendulumcart')))

    from pendulumcart import PendulumCartEnv

    # Create environment
    env = PendulumCartEnv(dt=0.01, max_steps=300)

    # Create IHDP agent
    agent = IHDPAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        gamma=0.99,
        forgetting_factor=0.99,
        initial_covariance=1.0,
        hidden_sizes={'actor': [6, 6], 'critic': [6, 6]},
        learning_rates={'actor': 5e-2, 'critic': 1e-1}
    )

    # Get a sample observation and transition for gradient checking
    obs, _ = env.reset()
    action = agent.get_action(obs)
    next_obs, reward, _, _, _ = env.step(action)

    print("Running gradient checking on IHDP agent...\n")

    # Perform gradient checking
    results = GradientChecker.check_agent_gradients(
        agent=agent,
        obs=obs,
        next_obs=next_obs,
        reward=reward,
        epsilon=1e-5,
        tolerance=1e-3
    )

    # Print results
    GradientChecker.print_gradient_check_results(results)

    # Access individual results if needed
    print("Detailed Results:")
    print(f"Actor passed: {results['actor']['passed']}")
    print(f"Critic passed: {results['critic']['passed']}")
    print(f"All checks passed: {results['all_passed']}")

    if results['all_passed']:
        print("\nSUCCESS: Gradient implementation is correct!")
    else:
        print("\nNote:")
        if results['actor']['num_checks'] == 0:
            print("  - Actor network has 0 checks (expected: uses manual gradient via RLS Jacobian)")
        else:
            if not results['actor']['passed']:
                print(f"  - Actor max error: {results['actor']['max_error']:.2e}")
        if not results['critic']['passed']:
            print(f"  - Critic max error: {results['critic']['max_error']:.2e}")
        else:
            print(f"  - Critic network PASSED with max error: {results['critic']['max_error']:.2e}")


if __name__ == "__main__":
    main()
