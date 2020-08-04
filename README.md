*photontorch is now open-source on GitHub: http://github.com/flaport/photontorch*

# photontorch_paper

Laporte, Floris, Joni Dambre, and Peter Bienstman. *"Highly parallel simulation
and optimization of photonic circuits in time and frequency domain based on the
deep-learning framework PyTorch."* Scientific reports 9.1 (2019): 5918.

# Photontorch has evolved...
The Photontorch API has evolved since the writing of this paper. Most of the notebooks won't run with the newest photontorch version. 
To run the notebooks in this repository, install photontorch via the `photontorch_paper` branch:
```
pip install git+https://github.com/flaport/photontorch.git@photontorch_paper
```

## Optimization Simulations
### CROW optimization
* [filter_design.ipynb](optimization/filter_design.ipynb): notebook that optimizes the CROW.
* [crow folder](optimization/crow): folder containing the losses during training and the final band-pass filter obtained byt the CROW optimization

### Reservoir optimization
* [xor_swirl.ipynb](optimization/xor_swirl.ipynb): notebook that optimizes a single reservoir to perform the XOR on bits in a bit stream.
* [xor_swirl_cascaded.ipynb](optimization/xor_swirl_cascaded.ipynb): notebook that optimizes a cascaded reservoir to perform the XOR on bits in a bit stream.
* [xor_swirl_plot.ipynb](optimization/xor_swirl_plot.ipynb): notebook that makes the plot used in the paper.
* [reservoir_losses](optimization/reservoir_losses): folder containing the learning curves and detected streams for the reservoir optimization.

### Unitary Matrix optimization
* [mnist_eunn_photontorch.ipynb](optimization/mnist_eunn_photontorch.ipynb): notebook that optimizes a unitary matrix network to perform the pixel-by-pixel MNIST digit recognition task (this optimization takes about 24 hours).
* [mnist_checkpoints/eunn_photontorch_capacity2/step_39999_acc=91.33_loss=0.30.pkl](optimization/mnist_checkpoints/eunn_photontorch_capacity2/step_39999_acc=91.33_loss=0.30.pkl): sample weights for the unitary matrix network implemented in Photontorch (num_hidden=256; capacity=2).


## Performance Simulations

### Visualization
* [FrequencyDomain.ipynb](performance/FrequencyDomain.ipynb): notebook containing performance visualizations for the frequency domain simulations
* [TimeDomain.ipynb](performance/TimeDomain.ipynb): notebook containing performance visualizations for the time domain simulations
* [Combined.ipynb](performance/Combined.ipynb): notebook containing performance visualizations for both the time domain simulations and the frequency domain simulations.

### Photontorch CROW
* [frequency_domain_single_wl_num_rings_sweep.ipynb](performance/PhotontorchCrow/frequency_domain_single_wl_num_rings_sweep/frequency_domain_single_wl_num_rings_sweep.ipynb): notebook measuring the time of simulation of a CROW with Photontorch in the frequency domain
* [frequency_domain_single_wl_num_rings_sweep.csv](performance/PhotontorchCrow/frequency_domain_single_wl_num_rings_sweep/frequency_domain_single_wl_num_rings_sweep.csv): recorded times for the simulation of a CROW in Photontorch in the frequency domain.
* [frequency_domain_single_wl_num_rings_sweep_cuda.csv](performance/PhotontorchCrow/frequency_domain_single_wl_num_rings_sweep/frequency_domain_single_wl_num_rings_sweep_cuda.csv): recorded times for the simulation of a CROW with Photontorch in the frequency domain using a GPU.
* [time_domain_wl_sweep_num_rings_sweep.ipynb](performance/PhotontorchCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_wl_sweep_num_rings_sweep.ipynb): notebook measuring the time of simulation of a CROW with Photontorch in the time domain for varying number of wavelengths
* [time_domain_3000_wl_sweep_num_rings_sweep.csv](performance/PhotontorchCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_3000_wl_sweep_num_rings_sweep.csv): recorded times for the simulation of a CROW with Photontorch in the time domain for multiple wavelengths.
* [time_domain_3000_wl_sweep_num_rings_sweep_cuda.csv](performance/PhotontorchCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_3000_wl_sweep_num_rings_sweep_cuda.csv): recorded times for the simulation of a CROW with Photontorch in the time domain for multiple wavelengths.
* [time_domain_batch_sweep_num_rings_sweep.ipynb](performance/PhotontorchCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_batch_sweep_num_rings_sweep.ipynb): notebook measuring the time of simulation of a CROW with Photontorch in the time domain for varying number of wavelengths
* [time_domain_3000_batch_sweep_num_rings_sweep.csv](performance/PhotontorchCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_3000_batch_sweep_num_rings_sweep.csv): recorded times for the simulation of a CROW with Photontorch in the time domain for multiple wavelengths.
* [time_domain_3000_batch_sweep_num_rings_sweep_cuda.csv](performance/PhotontorchCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_3000_batch_sweep_num_rings_sweep_cuda.csv): recorded times for the simulation of a CROW with Photontorch in the time domain for multiple wavelengths.

### Caphe CROW
* [frequency_domain_single_wl_num_rings_sweep.ipynb](performance/CapheCrow/frequency_domain_single_wl_num_rings_sweep/frequency_domain_single_wl_num_rings_sweep.ipynb): notebook measuring the time of simulation of a CROW with Caphe in the frequency domain
* [frequency_domain_single_wl_num_rings_sweep.csv](performance/CapheCrow/frequency_domain_single_wl_num_rings_sweep/frequency_domain_single_wl_num_rings_sweep.csv): recorded times for the simulation of a CROW with Caphe in the frequency domain.
* [time_domain_wl_sweep_num_rings_sweep.ipynb](performance/CapheCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_wl_sweep_num_rings_sweep.ipynb): notebook measuring the time of simulation of a CROW with Caphe in the time domain for varying number of wavelengths
* [time_domain_3000_wl_sweep_num_rings_sweep.csv](performance/CapheCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_3000_wl_sweep_num_rings_sweep.csv): recorded times for the simulation of a CROW with Caphe in the time domain for multiple wavelengths.

### Interconnect CROW
* [sample_crow_511_rings.icp](performance/InterconnectCrow/sample_crow_511_rings.icp): sample interconnect CROW simulation file (in this case with 511 rings) used for performance measurements.
* [frequency_domain_single_wl_num_rings_sweep.csv](performance/InterconnectCrow/frequency_domain_single_wl_num_rings_sweep/frequency_domain_single_wl_num_rings_sweep.csv): recorded times for the simulation of a CROW with Interconnect in the frequency domain.
* [time_domain_3000_wl_sweep_num_rings_sweep.csv](performance/InterconnectCrow/time_domain_wl_sweep_num_rings_sweep/time_domain_3000_wl_sweep_num_rings_sweep.csv): recorded times for the simulation of a CROW with Interconnect in the time domain for multiple wavelengths.
