from Simple-Neural-Network-Design import Node, Layer, NeuralNetwork

#Building the Layer Sizes
input_size = 8
hidden_size1 = 12
output_size = 2

#Reading specific file data using CSV and cleaning it however you like. Put the data into a numpy array.
data = pd.read_csv("file_path")

#Change network learning paramaters to whatever you like
nn = NeuralNetwork(input_size, hidden_size1, output_size)
losses = nn.train(input_data, target_data, epochs=500, learning_rate=0.01)
