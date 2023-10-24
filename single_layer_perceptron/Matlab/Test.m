clc;
clear;
close all;

% Define input data and target for XOR
x_xor = [1, 1, 0, 0; 1, 0, 1, 0; 1, 1, 1, 1];
y_xor = [0, 1, 1, 0];

% Define input data and target for AND
x_and = [1, 1, 0, 0; 1, 0, 1, 0; 1, 1, 1, 1];
y_and = [1, 0, 0, 0];

% Define input data and target for OR
x_or = [1, 1, 0, 0; 1, 0, 1, 0; 1, 1, 1, 1];
y_or = [1, 1, 1, 0];

% Create Perceptron models for XOR, AND, and OR
learning_rate = 0.01;
n_iteration = 1000;
n_features = 2;
model_xor = Perceptron(n_features, learning_rate, n_iteration);
model_and = Perceptron(n_features, learning_rate, n_iteration);
model_or = Perceptron(n_features, learning_rate, n_iteration);

% Train the Perceptron models
model_xor.train(x_xor, y_xor);
model_and.train(x_and, y_and);
model_or.train(x_or, y_or);

% Display results for XOR
disp('XOR');
disp('The weight of the perceptron is');
disp(model_xor.weight);
disp('The output of the perceptron is');
disp(model_xor.predict(x_xor) >= 0.5);

% Display results for AND
disp('AND');
disp('The weight of the perceptron is');
disp(model_and.weight);
disp('The output of the perceptron is');
disp(model_and.predict(x_and) >= 0.5);

% Display results for OR
disp('OR');
disp('The weight of the perceptron is');
disp(model_or.weight);
disp('The output of the perceptron is');
disp(model_or.predict(x_or) >= 0.5);
