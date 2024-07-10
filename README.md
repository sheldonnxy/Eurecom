
# Thesis Code

## Part1: Environment setup, dataset generation, visualization

1. Utilize the Conda to create a virtual environment with Python 3.7.16 and install all the dependencies.
2. Use RM_Create.py and DataSampler.py to generate RadioMap and related datasets. 
3. All files are saved under the Data directory accordingly.


## Part2: UNet setup, training, results saving
4. Load the dataset with input as the RMs and Masked measurement maps, then start the training process for the UNet.
5. Save the Uncertainty Map, Predicted Map, and the trained UNet, under the Training directory accordingly.
6. An example of the predicted RadioMap could be found in the results folder.


## Part3: MARL setting
7. Set the agent(s) and environment (state space, action space, reward, etc.) for test and deploy different MARL algorithms as Multi-agent PPO and I2Q
8. Start the training process, let the agent(s) optimize the trajectory by maximizing the reward. 
9. Note: as there are 20 measurements in the data generation process, we take 3 agents with 6 consecutive measurements each, i.e. 18 measurements totally.

## Part4: MARL evaluation
10. Input the measurements from the optimized trajectory to the trained UNet for RM prediction. The accuracies can be slightly improved, shown in the plots.
11. Extension: MARL agents could improve the prediction results of the UNet, also the optimized trajectory could help the UNet training process backwards (TBD)
12. Advantage: less measurements, better accuracy, etc.
