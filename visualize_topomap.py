import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from torch.utils.data import DataLoader, TensorDataset
import os
# Ensure that EEGNet is exported from your model_attention.py file
from AttentionEEGNet import EEGNet

# --- 1. Parameters Setup ---
CHANS = 23 # Number of EEG Channels (CHB-MIT)
SAMPLES = 512
# Setting device to CPU for safe plotting, MNE plotting functions are CPU-bound anyway
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Common 23 channel names for CHB-MIT dataset
CH_NAMES_23 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'T9', 'T10', 'FC6', 'FC5'
]

# --- 2. Helper Function: Extract Attention Weights for Topomap ---
def get_channel_weights_for_topomap(model, data_loader, F1, D):
    """
    Loads the model, runs a batch of seizure data, and extracts weights 
    suitable for Topomap visualization.
    
    The SE Block weights (F1*D) are combined with the spatial filter weights (23 channels) 
    from Block 2 (Depthwise Convolution) to obtain a final score for each of the 23 electrodes.
    
    Args:
        model (nn.Module): The trained Attention-EEGNet model.
        data_loader (DataLoader): DataLoader for test data.
        F1 (int): Number of temporal filters in Block 1 (e.g., 8).
        D (int): Depth multiplier for Depthwise Convolution (e.g., 2).
        
    Returns:
        np.array: Attention score for the 23 channels (1D array).
    """
    model.eval()
    
    # Get the spatial filter weights from Block 2 (conv2_spatial)
    # Weight shape: (F1 * D, F1, Chans, 1) -> e.g., (16, 8, 23, 1)
    # Ensure model.conv2_spatial is correctly accessed as a sequential block
    spatial_filters = model.conv2_spatial[0].weight.data.cpu().numpy()
    
    all_channel_scores = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Select only one seizure sample (label is 1)
            seizure_indices = (labels == 1).nonzero(as_tuple=True)[0]
            if len(seizure_indices) == 0:
                continue

            # Pick the first seizure segment
            sample_input = inputs[seizure_indices[0]].unsqueeze(0).to(DEVICE)
            
            # Run the model (returns classification result and F1*D attention weights)
            # attention_map shape: (1, F1*D) -> e.g., (1, 16)
            _, attention_map = model(sample_input)
            
            # Convert attention tensor to numpy array and reshape to (F1*D, 1, 1, 1)
            attention_weights_tensor = attention_map.squeeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3).cpu().numpy()
            # Shape: (F1*D, 1, 1, 1) -> e.g., (16, 1, 1, 1)

            # Weight combination: Multiply the spatial filters by the attention weights.
            # (F1*D, F1, 23, 1) * (F1*D, 1, 1, 1) -> Broadcast multiplication, result shape (F1*D, F1, 23, 1)
            weighted_filters = spatial_filters * attention_weights_tensor
            
            # Sum and average to get the final score for the 23 channels
            # Average over F1*D (Attention Features) and F1 (Temporal Filters) dimensions, 
            # leaving the 23 channel scores.
            channel_scores = np.mean(weighted_filters, axis=(0, 1, 3)) # Shape: (23,)
            
            all_channel_scores.append(channel_scores)
            
            # Only process one sample for visualization
            break

    if all_channel_scores:
        return all_channel_scores[0]
    else:
        print("No seizure sample found or model returned incorrect weights. Check test set labels.")
        return np.zeros(CHANS)

# --- 3. Main Plotting Function ---
def main_visualize():
    # Default model hyperparameters (must match defaults in model.py)
    F1, D = 8, 2 
    
    # 1. Load the Model
    model = EEGNet(nb_classes=2, Chans=CHANS, Samples=SAMPLES, F1=F1, D=D).to(DEVICE)
    model_path = './best_eegnet_model_attention.pth'
    if not os.path.exists(model_path):
        print(f"Error: Best model file '{model_path}' not found. Please train Attention-EEGNet first.")
        return
        
    try:
        # Load weights: crucial for visualization
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load model weights. Ensure model.py and the saved .pth file structure match: {e}")
        return
    
    # 2. Simulate Loading Test Data
    data_path = './data/'
    try:
        # Assuming data files exist
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        Y_test = np.load(os.path.join(data_path, 'Y_test.npy'))
        X_test_tensor = torch.from_numpy(X_test).to(torch.float32).unsqueeze(1)
        Y_test_tensor = torch.from_numpy(Y_test).to(torch.long)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    except FileNotFoundError:
        print(f"Error: Please ensure data files X_test.npy and Y_test.npy are in the '{data_path}' directory.")
        return

    # 3. Extract Attention Weights
    print("\n--- Extracting Attention Weights and Mapping to 23 Electrodes ---")
    weights = get_channel_weights_for_topomap(model, test_loader, F1, D)
    
    if weights.sum() == 0 and not np.any(weights):
        print("Could not extract valid weights, exiting plotting.")
        return

    # 4. Get Electrode Locations (using standard 10-20 system)
    info = mne.create_info(ch_names=CH_NAMES_23, sfreq=256, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # 5. Plotting
    print("\n--- Drawing Topomap ---")
    fig, ax = plt.subplots(figsize=(7, 7))
    
    abs_weights = np.abs(weights)
    vmax = abs_weights.max() if abs_weights.max() > 0 else 1.0 
    
    mne.viz.plot_topomap(
        data=weights, 
        pos=info, 
        names=info.ch_names, 
        cmap='RdBu_r', 
        sensors=True, 
        axes=ax, 
        vlim=(-vmax, vmax),
        sphere=0.1
    )
    ax.set_title("Channel Attention Score for Seizure Localization (Topomap)", fontsize=14, pad=20)
    
    # 6. Save and Show the Plot
    plt.savefig('./topomap_localization.png', dpi=300)
    plt.show() 
    print(f"\n✅ Critical paper figure generated and saved as: topomap_localization.png")
    
if __name__ == "__main__":
    main_visualize()