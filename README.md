# Neural-Network
1. Read Introduction to Neural Network.pdf
2. Open neuralnet.py
3. 9 parameters in command line: <train input> <validation input> <train out> <validation out> <metrics out> <num epoch> <hidden units> <init flag> <learning rate>. Sample input: $ python neuralnet.py smalltrain.csv smallvalidation.csv model1train_out.labels model1val_out.labels model1metrics_out.txt 5 4 2 0.1. You will get the following result in model1metrics_out.txt: epoch=1 crossentropy(train): 2.18506276114 epoch=1 crossentropy(validation): 2.18827302588 epoch=2 crossentropy(train): 1.90103257727 epoch=2 crossentropy(validation): 1.91363803461 error(train): 0.728 error(validation): 0.77.
4. You may also try big datasets. It may take 20 minutes to train and validate LargeTrain.csv and LargeValidation.csv
