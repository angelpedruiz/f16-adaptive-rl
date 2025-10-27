"""
Test script for RLSModel - Testing it as a state estimator on dummy data
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '..')
from ihdp import RLSModel


def test_rls_model_simple_linear_system():
    """Test RLSModel on a simple linear system where ground truth is known"""
    print("=" * 70)
    print("Test 1: Simple Linear System (Ground Truth Known)")
    print("=" * 70)

    # Define a simple linear system: x_{t+1} = 0.9*x_t + 0.5*u_t
    obs_dim = 1
    act_dim = 1

    # Ground truth jacobians
    F_true = np.array([[0.9]])  # state transition
    G_true = np.array([[0.5]])  # control sensitivity

    # Initialize RLS model
    model = RLSModel(obs_dim, act_dim, forgetting_factor=0.99, delta=1.0)

    # Generate training data
    np.random.seed(42)
    n_samples = 100
    x_current = np.array([1.0])
    states = [x_current.copy()]
    actions = []

    print(f"\nTraining RLS model on {n_samples} samples...")
    for i in range(n_samples):
        u = np.random.randn(act_dim) * 0.5  # random action
        # True next state
        x_next = F_true @ x_current + G_true @ u
        # Add small noise
        x_next += np.random.randn(obs_dim) * 0.01

        # Update model
        model.update(x_current, u, x_next)
        x_current = x_next
        states.append(x_current.copy())
        actions.append(u.copy())

    # Check learned parameters
    F_learned, G_learned = model.get_jacobians()
    print(f"\nGround truth F: {F_true}")
    print(f"Learned F:      {F_learned}")
    print(f"F error:        {np.linalg.norm(F_learned - F_true):.6f}")

    print(f"\nGround truth G: {G_true}")
    print(f"Learned G:      {G_learned}")
    print(f"G error:        {np.linalg.norm(G_learned - G_true):.6f}")

    # Test prediction on new data
    print(f"\n--- Testing predictions ---")
    test_states = [np.array([0.0]), np.array([1.0]), np.array([-0.5])]
    test_actions = [np.array([0.2]), np.array([-0.3]), np.array([0.5])]
    test_errors = []

    for x_test, u_test in zip(test_states, test_actions):
        # True next state
        x_true_next = F_true @ x_test + G_true @ u_test
        # Predicted next state
        x_pred_next = model.predict(x_test, u_test)
        error = np.linalg.norm(x_pred_next - x_true_next)
        test_errors.append(error)

        print(f"State: {x_test[0]:6.2f}, Action: {u_test[0]:6.2f}")
        print(f"  True next:     {x_true_next[0]:8.4f}")
        print(f"  Predicted:     {x_pred_next[0]:8.4f}")
        print(f"  Error:         {error:8.6f}\n")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Test 1: Simple 1D Linear System', fontsize=14, fontweight='bold')

    # Plot training trajectory
    ax = axes[0, 0]
    ax.plot(states, label='Training trajectory', marker='o', markersize=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('State')
    ax.set_title('Training Trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot learned vs true parameters
    ax = axes[0, 1]
    params = ['F', 'G']
    true_vals = [F_true[0, 0], G_true[0, 0]]
    learned_vals = [F_learned[0, 0], G_learned[0, 0]]
    x_pos = np.arange(len(params))
    width = 0.35
    ax.bar(x_pos - width/2, true_vals, width, label='True', alpha=0.8)
    ax.bar(x_pos + width/2, learned_vals, width, label='Learned', alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('Learned vs True Parameters')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot test predictions
    ax = axes[1, 0]
    test_x_vals = [x[0] for x in test_states]
    test_u_vals = [u[0] for u in test_actions]
    test_true_next = [F_true[0, 0] * x + G_true[0, 0] * u for x, u in zip(test_x_vals, test_u_vals)]
    test_pred_next = [model.predict(x, u)[0] for x, u in zip(test_states, test_actions)]

    x_pos = np.arange(len(test_states))
    ax.bar(x_pos - width/2, test_true_next, width, label='True', alpha=0.8)
    ax.bar(x_pos + width/2, test_pred_next, width, label='Predicted', alpha=0.8)
    ax.set_ylabel('Next State')
    ax.set_title('Test Predictions')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Test {i+1}' for i in range(len(test_states))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot prediction errors
    ax = axes[1, 1]
    ax.bar(range(len(test_errors)), test_errors, alpha=0.8, color='orange')
    ax.axhline(y=np.mean(test_errors), color='r', linestyle='--', label=f'Mean: {np.mean(test_errors):.6f}')
    ax.set_ylabel('Prediction Error')
    ax.set_xlabel('Test Sample')
    ax.set_title('Prediction Errors on Test Data')
    ax.set_xticks(range(len(test_errors)))
    ax.set_xticklabels([f'Test {i+1}' for i in range(len(test_errors))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('test1_simple_linear_system.png', dpi=150, bbox_inches='tight')
    print("Saved plot: test1_simple_linear_system.png")
    plt.close()


def test_rls_model_multidim_system():
    """Test RLSModel on a 2D system"""
    print("\n" + "=" * 70)
    print("Test 2: 2D System")
    print("=" * 70)

    obs_dim = 2
    act_dim = 2

    # Define a simple 2D linear system
    F_true = np.array([[0.8, 0.1],
                       [0.0, 0.9]])
    G_true = np.array([[0.5, 0.2],
                       [0.1, 0.6]])

    model = RLSModel(obs_dim, act_dim, forgetting_factor=0.99, delta=1.0)

    # Generate training data
    np.random.seed(42)
    n_samples = 200
    x_current = np.array([1.0, 0.5])
    trajectory = [x_current.copy()]

    print(f"\nTraining RLS model on {n_samples} samples...")
    for i in range(n_samples):
        u = np.random.randn(act_dim) * 0.3
        x_next = F_true @ x_current + G_true @ u
        x_next += np.random.randn(obs_dim) * 0.01
        model.update(x_current, u, x_next)
        x_current = x_next
        trajectory.append(x_current.copy())

    trajectory = np.array(trajectory)
    F_learned, G_learned = model.get_jacobians()

    print(f"\nGround truth F:")
    print(F_true)
    print(f"\nLearned F:")
    print(F_learned)
    print(f"F error: {np.linalg.norm(F_learned - F_true):.6f}")

    print(f"\nGround truth G:")
    print(G_true)
    print(f"\nLearned G:")
    print(G_learned)
    print(f"G error: {np.linalg.norm(G_learned - G_true):.6f}")

    # Test predictions
    print(f"\n--- Testing predictions ---")
    test_errors = []
    for _ in range(3):
        x_test = np.random.randn(obs_dim) * 0.5
        u_test = np.random.randn(act_dim) * 0.3

        x_true_next = F_true @ x_test + G_true @ u_test
        x_pred_next = model.predict(x_test, u_test)
        error = np.linalg.norm(x_pred_next - x_true_next)
        test_errors.append(error)

        print(f"State: {x_test}, Action: {u_test}")
        print(f"  True next: {x_true_next}")
        print(f"  Predicted: {x_pred_next}")
        print(f"  Error:     {error:.6f}\n")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Test 2: 2D Linear System', fontsize=14, fontweight='bold')

    # Plot 2D trajectory
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=3, alpha=0.7, label='Training trajectory')
    ax.scatter([1.0], [0.5], color='green', s=100, label='Start', zorder=5)
    ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], color='red', s=100, label='End', zorder=5)
    ax.set_xlabel('State x1')
    ax.set_ylabel('State x2')
    ax.set_title('2D Training Trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot state components over time
    ax = axes[0, 1]
    ax.plot(trajectory[:, 0], label='x1', marker='o', markersize=3, alpha=0.7)
    ax.plot(trajectory[:, 1], label='x2', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('State Value')
    ax.set_title('State Components Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot F and G matrix errors
    ax = axes[1, 0]
    F_error = np.abs(F_learned - F_true)
    G_error = np.abs(G_learned - G_true)
    im = ax.imshow(np.vstack([F_error, G_error]), cmap='RdYlGn_r', aspect='auto')
    ax.set_ylabel('Matrix Element')
    ax.set_xlabel('Column')
    ax.set_title('Absolute Parameter Errors (F above, G below)')
    row_labels = [f'F[{i%2},{j%2}]' for i in range(2) for j in range(2)] + \
                 [f'G[{i%2},{j%2}]' for i in range(2) for j in range(2)]
    ax.set_yticks(range(8))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax)

    # Plot test errors
    ax = axes[1, 1]
    ax.bar(range(len(test_errors)), test_errors, alpha=0.8, color='orange')
    ax.axhline(y=np.mean(test_errors), color='r', linestyle='--',
               label=f'Mean: {np.mean(test_errors):.6f}')
    ax.set_ylabel('Prediction Error')
    ax.set_xlabel('Test Sample')
    ax.set_title('Prediction Errors on Test Data')
    ax.set_xticks(range(len(test_errors)))
    ax.set_xticklabels([f'Test {i+1}' for i in range(len(test_errors))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('test2_2d_system.png', dpi=150, bbox_inches='tight')
    print("Saved plot: test2_2d_system.png")
    plt.close()


def test_rls_model_random_walk():
    """Test RLSModel on a random walk with control"""
    print("\n" + "=" * 70)
    print("Test 3: Random Walk with Control (3D)")
    print("=" * 70)

    obs_dim = 3
    act_dim = 2

    model = RLSModel(obs_dim, act_dim, forgetting_factor=0.98, delta=2.0)

    # Generate training data from random walk
    np.random.seed(42)
    n_samples = 300
    x_current = np.array([0.0, 0.0, 0.0])
    trajectory = [x_current.copy()]

    print(f"\nTraining RLS model on {n_samples} random walk samples...")
    errors = []
    for i in range(n_samples):
        u = np.random.randn(act_dim) * 0.2
        # Simulate as a random walk with control influence
        x_next = x_current + 0.05 * np.random.randn(obs_dim) + np.concatenate([0.1 * u, [0.0]])

        model.update(x_current, u, x_next)
        x_current = x_next
        trajectory.append(x_current.copy())

    trajectory = np.array(trajectory)
    F_learned, G_learned = model.get_jacobians()

    print(f"\nLearned F (state transition):")
    print(F_learned)
    print(f"\nLearned G (control sensitivity):")
    print(G_learned)

    # Test predictions
    print(f"\n--- Testing predictions on new data ---")
    test_errors = []
    for _ in range(5):
        x_test = np.random.randn(obs_dim) * 0.5
        u_test = np.random.randn(act_dim) * 0.2

        # True next state (from the same random walk model)
        x_true_next = x_test + 0.05 * np.random.randn(obs_dim) + np.concatenate([0.1 * u_test, [0.0]])
        x_pred_next = model.predict(x_test, u_test)
        error = np.linalg.norm(x_pred_next - x_true_next)
        test_errors.append(error)

        print(f"Test {_+1}: Error = {error:.6f}")

    print(f"\nMean prediction error: {np.mean(test_errors):.6f}")
    print(f"Max prediction error:  {np.max(test_errors):.6f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Test 3: 3D Random Walk with Control', fontsize=14, fontweight='bold')

    # Plot 3D trajectory components
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], label='x1', marker='o', markersize=2, alpha=0.7)
    ax.plot(trajectory[:, 1], label='x2', marker='s', markersize=2, alpha=0.7)
    ax.plot(trajectory[:, 2], label='x3', marker='^', markersize=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('State Value')
    ax.set_title('State Components Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot F matrix heatmap
    ax = axes[0, 1]
    im = ax.imshow(F_learned, cmap='coolwarm', aspect='auto')
    ax.set_title('Learned F Matrix (State Transition)')
    ax.set_xlabel('Column (Previous State)')
    ax.set_ylabel('Row (Next State Dimension)')
    plt.colorbar(im, ax=ax)

    # Plot G matrix heatmap
    ax = axes[1, 0]
    im = ax.imshow(G_learned, cmap='coolwarm', aspect='auto')
    ax.set_title('Learned G Matrix (Control Sensitivity)')
    ax.set_xlabel('Column (Action)')
    ax.set_ylabel('Row (State Dimension)')
    plt.colorbar(im, ax=ax)

    # Plot test prediction errors
    ax = axes[1, 1]
    ax.bar(range(len(test_errors)), test_errors, alpha=0.8, color='steelblue')
    ax.axhline(y=np.mean(test_errors), color='r', linestyle='--',
               label=f'Mean: {np.mean(test_errors):.6f}')
    ax.axhline(y=np.max(test_errors), color='orange', linestyle=':', alpha=0.7,
               label=f'Max: {np.max(test_errors):.6f}')
    ax.set_ylabel('Prediction Error')
    ax.set_xlabel('Test Sample')
    ax.set_title('Prediction Errors on Test Data')
    ax.set_xticks(range(len(test_errors)))
    ax.set_xticklabels([f'Test {i+1}' for i in range(len(test_errors))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('test3_random_walk.png', dpi=150, bbox_inches='tight')
    print("Saved plot: test3_random_walk.png")
    plt.close()


def test_rls_model_integration():
    """Integration test: continuous prediction"""
    print("\n" + "=" * 70)
    print("Test 4: Integration Test - Continuous Prediction")
    print("=" * 70)

    obs_dim = 2
    act_dim = 1

    # Define true system
    F_true = np.array([[0.95, 0.05], [0.0, 0.9]])
    G_true = np.array([[0.3], [0.5]])

    model = RLSModel(obs_dim, act_dim, forgetting_factor=0.99, delta=1.0)

    # Train model
    np.random.seed(42)
    n_train = 150
    x_current = np.array([1.0, 0.5])

    print(f"Training on {n_train} samples...")
    for _ in range(n_train):
        u = np.random.randn(act_dim) * 0.3
        x_next = F_true @ x_current + G_true @ u
        x_next += np.random.randn(obs_dim) * 0.01
        model.update(x_current, u, x_next)
        x_current = x_next

    F_learned, G_learned = model.get_jacobians()
    print(f"F error: {np.linalg.norm(F_learned - F_true):.6f}")
    print(f"G error: {np.linalg.norm(G_learned - G_true):.6f}")

    # Continuous prediction: follow the predicted trajectory
    print(f"\n--- Predicting 20-step trajectory ---")
    x_init = np.array([1.0, 0.0])
    actions = [np.array([0.2]), np.array([-0.2]), np.array([0.1])] * 7  # cycle through 3 actions

    x_true = x_init.copy()
    x_pred = x_init.copy()

    true_trajectory = [x_true.copy()]
    pred_trajectory = [x_pred.copy()]
    errors = []
    action_sequence = []

    print(f"Initial state: {x_init}")

    for step, u in enumerate(actions):
        # True trajectory
        x_true = F_true @ x_true + G_true @ u

        # Predicted trajectory
        x_pred = model.predict(x_pred, u)

        error = np.linalg.norm(x_pred - x_true)
        errors.append(error)
        true_trajectory.append(x_true.copy())
        pred_trajectory.append(x_pred.copy())
        action_sequence.append(u[0])

        print(f"Step {step+1:2d}: u={u[0]:6.2f} | Error={error:.6f} | Pred={x_pred} | True={x_true}")

    # Convert to numpy arrays for plotting
    true_trajectory = np.array(true_trajectory)
    pred_trajectory = np.array(pred_trajectory)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Test 4: Integration Test - Continuous Prediction', fontsize=14, fontweight='bold')

    # Plot 2D trajectories
    ax = axes[0, 0]
    ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'o-', label='True trajectory', linewidth=2, markersize=4)
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 's--', label='Predicted trajectory', linewidth=2, markersize=4)
    ax.scatter([x_init[0]], [x_init[1]], color='green', s=150, marker='*', label='Start', zorder=5)
    ax.scatter([true_trajectory[-1, 0]], [true_trajectory[-1, 1]], color='red', s=100, marker='X', label='True End', zorder=5)
    ax.scatter([pred_trajectory[-1, 0]], [pred_trajectory[-1, 1]], color='orange', s=100, marker='X', label='Pred End', zorder=5)
    ax.set_xlabel('State x1')
    ax.set_ylabel('State x2')
    ax.set_title('2D Trajectory: True vs Predicted')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot state components over time
    ax = axes[0, 1]
    steps = range(len(true_trajectory))
    ax.plot(steps, true_trajectory[:, 0], 'o-', label='True x1', linewidth=2)
    ax.plot(steps, pred_trajectory[:, 0], 's--', label='Pred x1', linewidth=2)
    ax.plot(steps, true_trajectory[:, 1], '^-', label='True x2', linewidth=2)
    ax.plot(steps, pred_trajectory[:, 1], 'v--', label='Pred x2', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('State Value')
    ax.set_title('State Components Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    # Plot cumulative error over trajectory
    ax = axes[1, 0]
    cumulative_error = np.cumsum(errors)
    ax.plot(range(1, len(errors) + 1), errors, 'o-', label='Step error', linewidth=2, markersize=4)
    ax.plot(range(1, len(errors) + 1), cumulative_error, 's--', label='Cumulative error', linewidth=2, markersize=4)
    ax.axhline(y=np.mean(errors), color='r', linestyle=':', alpha=0.7, label=f'Mean: {np.mean(errors):.6f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Error')
    ax.set_title('Prediction Errors Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot actions and error relationship
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.bar(range(len(action_sequence)), action_sequence, alpha=0.6, color='steelblue', label='Actions')
    ax2.plot(range(len(errors)), errors, 'ro-', linewidth=2, markersize=5, label='Errors')
    ax.set_xlabel('Step')
    ax.set_ylabel('Action Value', color='steelblue')
    ax2.set_ylabel('Prediction Error', color='red')
    ax.set_title('Actions vs Prediction Errors')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('test4_integration.png', dpi=150, bbox_inches='tight')
    print("Saved plot: test4_integration.png")
    plt.close()


if __name__ == "__main__":
    test_rls_model_simple_linear_system()
    test_rls_model_multidim_system()
    test_rls_model_random_walk()
    test_rls_model_integration()

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
