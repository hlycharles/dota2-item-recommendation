# dota2-item-recommendation
A neural network that recommends items to purchase next based on current game status.

## Workflow
1. Generate features from raw data: `python feature_generator <raw_input_folder> <features_folder>`
2. Split examples into train, dev and test sets: `python split_data.py <features_folder> <datasets_folder>`
3. Run the neural network: `python nn.py <args>`.

## Command Line Arguments
The command line arguments that `nn.py` accepts are:
1. **Debug mode**. If the value if truthy, then print debug messages including input data size, epochs, costs and accuracies after each epoch.
2. **Hidden layer sizes**. The sizes of hidden layers separated by comma. For examples, `200,10` would run a two hidden layer network where 200 and 10 are the hidden layer sizes.
3. **Id**. Used to specify output file nams.
4. **Datasets folder** Folder that includes data separated into train, dev, test sets. The output from `split_data.py` can be used here.
