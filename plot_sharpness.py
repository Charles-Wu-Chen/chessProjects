import matplotlib.pyplot as plt
import numpy as np
from functions import sharpnessLC0
from test_sharpness_lc0 import TestSharpnessLC0

def plot_sharpness():
    # Extract test cases from TestSharpnessLC0
    test_cases = [
        ([300, 400, 300], 1.3929),
        ([100, 800, 100], 0.2071),
        ([1, 998, 1], 0.0210),
        ([0, 1000, 0], 0.0118),
        ([600, 100, 300], 20.4901),
        ([200, 300, 500], 2.0814),
        ([1000, 0, 0], 4000000.0000),
        ([0, 0, 1000], 4000000.0000),
        ([463, 142, 395], 12.1146),
        ([476, 134, 389], 13.3397),
        ([218, 382, 400], 1.4125),
        ([430, 96, 474], 26.854)
    ]

    # Generate more W and L values
    num_points = 1000
    w_values = np.random.randint(0, 1001, num_points)
    l_values = np.random.randint(0, 1001 - w_values)
    
    # Filter out cases where W + L >= 1000
    valid_indices = w_values + l_values < 1000
    w_values = w_values[valid_indices]
    l_values = l_values[valid_indices]

    # Calculate sharpness for simulated values
    d_values = 1000 - w_values - l_values
    simulated_inputs = np.column_stack((w_values, d_values, l_values))
    simulated_outputs = [min(sharpnessLC0(input_case), 20) for input_case in simulated_inputs]

    # Extract original test case data
    original_w_values = [case[0][0]/1000 for case in test_cases]
    original_l_values = [case[0][2]/1000 for case in test_cases]
    expected_outputs = [min(case[1], 20) for case in test_cases]
    actual_outputs = [min(sharpnessLC0(case[0]), 20) for case in test_cases]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot simulated outputs
    scatter = ax.scatter(w_values/1000, l_values/1000, simulated_outputs, c=simulated_outputs, cmap='viridis', alpha=0.5, label='Simulated Output')

    # Plot original test cases
    ax.scatter(original_w_values, original_l_values, expected_outputs, c='r', marker='o', s=100, label='Expected Output (Test Cases)')
    ax.scatter(original_w_values, original_l_values, actual_outputs, c='b', marker='x', s=100, label='Actual Output (Test Cases)')

    # Add labels and title
    ax.set_xlabel('W (Win Probability)')
    ax.set_ylabel('L (Loss Probability)')
    ax.set_zlabel('Sharpness')
    ax.set_title('SharpnessLC0 Function: Simulated and Test Case Outputs (Capped at 20)')
    ax.legend()

    # Set z-axis limit to 20
    ax.set_zlim(0, 20)

    # Add a color bar
    cbar = plt.colorbar(scatter, label='Sharpness')
    cbar.set_ticks(np.linspace(0, 20, 5))

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_sharpness()