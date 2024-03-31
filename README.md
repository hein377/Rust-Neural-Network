# Rust-Neural-Network
**Member:** Hein Htut (hhtut2)
**Project Description:** I hope to implement a classical neural network, of user-defined size, from scratch in Rust. I've always been interested in machine learning and since neural networks are at the core of ML, I decided to focus on this topic. I am looking forward to seeing how membership will integrate with data storage and manipulation! If possible, I think it would it would also be interesting to implement parallelism in running different chunks of data through multiple neural networks and combining them later but I doubt I will reach this far in the project.

## Technical Overview
The project can be sorted into four main steps: preprocessing the dataset, creating/initializing the neural network structure, implementing forward propagation, and finally implementing back propagation.

### Processing the Dataset
The portion will be fairly straightforward, with a required input of a .csv file only containing quantitative data (with the exception of feature labels) with the last column being the class label. For now, datasets will also be required to not contain missing data fields, but if time permits, I will add the code to fill such missing fields with the averages of the rest of the feature's data. The .csv file will then be processed with pandas and turned into a numPy array.

### Creating/Initializing the NN Structure
Similar to the last step, this step will also be fairly simple, with random weights and bias values assigned into numpy matrices/vectors of user-defined dimensions (in the form of # of perceptions per layer).

### Implementing Forward Propagation
This will be the first big challenge of my project. I think my biggest issue will be dealing with vanishing/exploding gradients especially considering the dataset and the dimensions of the neural network will be provided by the user. One potential solution might be having the user also define the sequence of activation functions. However, this might pose additional problems in ensuring the output of one layer is suitable for the input of the next.

### Implementing Back Propagation
This will undoubtedly be the hardest part of the project as it requires great familiarity with how I have structured my neural network. Implementing gradient descent will be rather math-heavy with partial derivatives and long chain rules so I will have to take extreme care with organization and knowing how the data is stored (e.g. how to access specific components of my network). This portion will likely require the most debugging out of all the steps of the project.
