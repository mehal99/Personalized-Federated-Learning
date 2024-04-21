import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['test_avg_acc']

def load_npz(npz_path):
    data = np.load(npz_path)
    return data['val_accuracy']


def plot_accuracies(pfedhn_acc, pfedhn_pc_acc, fedavg_acc):
    epochs = range(len(pfedhn_acc))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, pfedhn_acc, label='pFedHN')
    plt.plot(epochs, pfedhn_pc_acc, label='pFedHN-PC')
    print(len(fedavg_acc))
    plt.plot(np.arange(0, 100), fedavg_acc, label='FedAvg')

    plt.title('Test Accuracy (%) {Clients: 50, Inner Epochs: 20}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Paths to the files
# pfedhn_path_10 = 'experiments\\plot_experiments\\10clients_20inner_epochs\\pfedhn_results_10_20_5000.json'
# pfedhn_path_50 = 'experiments\\plot_experiments\\50clients_20inner_epochs\\pfedhn_results_50_20_5000.json'
# pfedhn_pc_path_10 = 'experiments\\plot_experiments\\10clients_20inner_epochs\\pfedhn_pc_results_10_20_5000.json'
# pfedhn_pc_path_50 = 'experiments\\plot_experiments\\50clients_20inner_epochs\\pfedhn_pc_results_50_20_5000.json'
# fedavg_npz_path_10 = 'experiments\\plot_experiments\\10clients_20inner_epochs\\acc_cnn_niid_10_20.npz'
# fedavg_npz_path_50 = 'experiments\\plot_experiments\\50clients_20inner_epochs\\acc_cnn_niid_50_20.npz'

# # Load the accuracies
# pfedhn_acc_5 = load_results(pfedhn_path_10)
# pfedhn_acc_20 = load_results(pfedhn_path_50)
# pfedhn_pc_acc_5 = load_results(pfedhn_pc_path_10)
# pfedhn_pc_acc_20 = load_results(pfedhn_pc_path_50)
# fedavg_acc_5 = load_npz(fedavg_npz_path_10)
# fedavg_acc_20 = load_npz(fedavg_npz_path_50)

# epochs = range(len(pfedhn_acc_5))
# plt.plot(np.arange(0, 100), fedavg_acc_5, label='FedAvg - 10 clients', linestyle='-', color='blue')
# plt.plot(np.arange(0, 100), fedavg_acc_20, label='FedAvg - 50 clients', linestyle='--', color='blue')
# plt.plot(epochs, pfedhn_acc_5, label='pFedHN - 10 clients', linestyle='-', color='green')
# plt.plot(epochs, pfedhn_acc_20, label='pFedHN - 50 clients', linestyle='--', color='green')
# plt.plot(epochs, pfedhn_pc_acc_5, label='PFedHN-PC - 10 clients', linestyle='-', color='red')
# plt.plot(epochs, pfedhn_pc_acc_20, label='PFedHN-PC - 50 clients', linestyle='--', color='red')

# # Adding labels and title.
# plt.xlabel('Epochs')
# plt.ylabel('Test Accuracy')
# plt.title('Test Accuracy for 10 and 50 clients')

# # Adding legend.
# plt.legend()

# # Adding grid.
# plt.grid(True)

# # Show the plot.
# plt.show()


pfedhn_path= 'experiments\\plot_experiments\\50clients_20inner_epochs\\pfedhn_results_50_20_5000.json'
pfedhn_pc_path = 'experiments\\plot_experiments\\50clients_20inner_epochs\\pfedhn_pc_results_50_20_5000.json'
fedavg_npz_path = 'experiments\\plot_experiments\\50clients_20inner_epochs\\acc_cnn_niid_50_20.npz'

pfedhn_acc = load_results(pfedhn_path)
pfedhn_pc_acc = load_results(pfedhn_pc_path)
fedavg_acc = load_npz(fedavg_npz_path)

# Plot the accuracies
plot_accuracies(pfedhn_acc, pfedhn_pc_acc, fedavg_acc)
