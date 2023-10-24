classdef Perceptron < handle
    % Perceptron class
    
    properties
        n_features
        weight
        learning_rate
        n_iteration
        dw
    end
    
    methods
        function obj = Perceptron(n_features, learning_rate, n_iteration)
            % Initialize the perceptron
            obj.n_features = n_features;
            obj.weight = randn(n_features + 1, 1);
            obj.learning_rate = learning_rate;
            obj.n_iteration = n_iteration;
            obj.dw = 0;
        end
        
        function output = sigmoid(obj, x)
            % Return the sigmoid of x
            output = (2 ./ (1 + exp(-x))) - 1;
        end
        
        function result = forward(obj, x)
            % Return the output of the perceptron with input x and weight w
            result = obj.sigmoid(obj.weight' * x);
        end
        
        function grad_result = grad(obj, x, y)
            % Compute the gradient of the weight
            output = obj.forward(x);
            obj.dw = sum(-0.5 .* (y - output) .* (1 - output.^2) .* x, 2);
            grad_result = obj.dw;
        end
        
        function updated_weight = step(obj)
            % Update the weight of the perceptron with the gradient
            obj.weight = obj.weight - obj.learning_rate * obj.dw;
            updated_weight = obj.weight;
        end
        
        function train(obj, x, y)
            % Train the perceptron
            for i = 1:obj.n_iteration
                obj.grad(x, y);
                obj.step();
            end
        end
        
        function prediction = predict(obj, x)
            % Return the prediction of the perceptron
            prediction = obj.forward(x);
        end
    end
end
