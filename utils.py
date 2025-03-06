import numpy as np
import torch
import matplotlib.pyplot  as plt
import pandas as pd
from torchmetrics.regression import R2Score, MeanSquaredError
import torch.nn.functional as F

# from ray.air.examples.custom_trainer import train_dataset
# from scipy.ndimage import label


def convert_I_TC(Stretch):
    Stretch = Stretch.clone().detach()
    I1 = torch.square(Stretch) + 2.0 / Stretch
    I2 = 2 * Stretch + 1 / torch.square(Stretch)
    return I1, I2

def convert_I_shr(Stretch):
    Stretch = Stretch.clone().detach()
    I1 = I2 = 3 + torch.square(Stretch)
    return I1, I2

def convert_P_ten(input):
    # convert output from nn to the prediction of P
    dPsidI1, dPsidI2, Stretch = input
    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)

    minus = two * (dPsidI1 * one / torch.square(Stretch) + dPsidI2 * one / torch.pow(Stretch, 3))
    stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus

    return stress

def convert_P_shr(input):
    dPsidI1, dPsidI2, Stretch = input
    two = torch.tensor(2.0, dtype=torch.float32)
    stress = two * Stretch * (dPsidI1 + dPsidI2)

    return stress

def dataloader(path, arg, batch_size):
    dfs = pd.read_excel(path, sheet_name='Sheet1')

    if arg == "tension":
        P = dfs.iloc[3:, 1].dropna().astype(np.float64)[16:].values
        strain = dfs.iloc[3:, 0].dropna().astype(np.float64)[16:].values
    elif arg == "compression":
        P = dfs.iloc[3:, 1].dropna().astype(np.float64).values[:17]
        strain = dfs.iloc[3:, 0].dropna().astype(np.float64).values[:17]
    else:
        P = dfs.iloc[3:, 3].dropna().astype(np.float64).values
        strain = dfs.iloc[3:, 2].dropna().astype(np.float64).values

    size_dataset = len(P)
    # num_batches = int((size_dataset + 0.5) / batch_size)
    num_test = size_dataset // 4

    num_test_samples = num_test

    x = torch.tensor(strain).clone().detach()
    Y = torch.tensor(P, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x, Y)

    # test_indices = range(num_test_samples)
    # train_indices = range(num_test_samples, size_dataset)
    #
    # test_dataset = torch.utils.data.Subset(dataset, test_indices)
    # train_dataset = torch.utils.data.Subset(dataset, train_indices)
    train_dataset = dataset
    test_dataset = dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, x, Y

def pack_up(state_dict):
    # input value state_dict, the dict storing the value of learned parameters
    # return the packed value of weights as well as t
    pack = None
    for key, value in state_dict.items():
        if pack is None:
            pack = value
        elif key == "fc.weight":
            pack = torch.cat((pack, value.T), 1)
        elif key == "selfupdatingpara.a" or key == "selfupdatingpara.b":
            print(f"{key}: {value}")
        else:
            pack = torch.cat((pack, value), 0)
    return pack


def colored_plot(state_dict, stretch, stress_real, model, data_mode):
    """
    Plot the incremental contribution of each model component as colored layers.

    :param state_dict: dict, storing the learned parameters (e.g. for a fully connected layer)
    :param stretch: numpy array, storing the stretch values (x-axis)
    :param stress_real: numpy array, storing the real stress values for comparison
    :param model: a PyTorch model mapping stretch to predicted stress
    :param data_mode: bounce between tension, compression and shear
    :return: None (displays a figure)
    """
    n = state_dict['fc.weight'].shape[1]
    assert n == 18, "Expected 18 components in the model."

    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    original_weight = state_dict['fc.weight'].clone()
    cumulative_pred = np.zeros_like(stretch)  # Start with zero contribution

    for i in range(n):
        # Create a mask that accumulates the first (i+1) components
        mask = torch.zeros_like(original_weight)
        mask[:, :i + 1] = original_weight[:, :i + 1]
        state_dict['fc.weight'] = mask

        # Load the updated weights into the model
        model.load_state_dict(state_dict, strict=False)
        # Compute the prediction for the current mask
        predicted_stress = model(stretch).detach().numpy().flatten()

        # Fill the area between the previous cumulative prediction and the new prediction
        plt.fill_between(stretch, cumulative_pred, predicted_stress,
                         color=colors[i], alpha=0.6)

        # Update the cumulative prediction for the next iteration
        cumulative_pred = predicted_stress.copy()

    cumulative_pred = torch.tensor(cumulative_pred, dtype=torch.float32)
    r2_metric = R2Score()
    r2 = r2_metric(cumulative_pred, stress_real)
    mse_metric = MeanSquaredError()
    mse_loss = mse_metric(cumulative_pred, stress_real)

    # Plot the overall prediction and the real stress for reference
    plt.plot(stretch, cumulative_pred, color='black', linewidth=2.5, label='Overall Prediction')
    plt.plot(stretch, stress_real, 'k--', linewidth=2.5, label="Real Stress")
    plt.xlabel("Stretch")
    plt.ylabel("Stress")
    plt.title("Model Prediction" + data_mode)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.grid(True)
    plt.text(0.05, 0.9, f"R²: {r2:.6f}\nLoss: {mse_loss:.6f}", transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    # Restore the original state_dict weights
    state_dict['fc.weight'] = original_weight






