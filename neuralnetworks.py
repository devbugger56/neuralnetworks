import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, hidden_layers, outputs):
        self.inputs = inputs
        self.input_data = None
        self.hidden_layers = hidden_layers
        self.outputs = outputs
        self.output = None
        self.func_type = None
        self.loss = 999999

    def feed_forward(self):
        exec(f"self.output = np.dot(self.input_data, self.weights0) + self.biases0")
        for i in range(1, len(self.hidden_layers)):
            self.current_func = i - 1
            exec(f"""
if self.function{self.current_func} == 'relu':
    self.func_type = 'relu'
elif self.function{self.current_func} == 'softmax':
    self.func_type = 'softmax'
elif self.function{self.current_func} == 'sigmoid':
    self.func_type = 'sigmoid'
elif self.function{self.current_func} == 'tanh':
    self.func_type = 'tanh'
elif self.function{self.current_func} is None:
    self.func_type = None
            """)
            if self.func_type is None:
                pass
            elif self.func_type == "relu":
                self.output = np.maximum(0, self.output)
            elif self.func_type == "softmax":
                exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
                probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                self.output = probabilities
            elif self.func_type == "sigmoid":
                self.output = 1 / (1 + (-exp(self.output)))
            elif self.func_type == "tanh":
                self.output = (exp(self.output) - (-exp(self.output))) / (exp(self.output) + (-exp(self.output)))
            exec(f"self.output = np.dot(self.output, self.weights{i}) + self.biases{i}")
        self.current_func = len(self.hidden_layers) - 1
        exec(f"""  
if self.function{self.current_func} == 'relu':
    self.func_type = 'relu'
elif self.function{self.current_func} == 'softmax':
    self.func_type = 'softmax'
elif self.function{self.current_func} == 'sigmoid':
    self.func_type = 'sigmoid'
elif self.function{self.current_func} == 'tanh':
    self.func_type = 'tanh'
elif self.function{self.current_func} is None:
    self.func_type = None
        """)
        if self.func_type is None:
            pass
        elif self.func_type == "relu":
            self.output = np.maximum(0, self.output)
        elif self.func_type == "softmax":
            exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probabilities
        elif self.func_type == "sigmoid":
            self.output = 1 / (1 + (-exp(self.output)))
        elif self.func_type == "tanh":
            self.output = (exp(self.output) - (-exp(self.output))) / (exp(self.output) + (-exp(self.output)))

    def back_propagation(self, solution):
        y_pred = self.output
        y_true = solution
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        data_loss = np.mean(negative_log_likelihoods)
        self.loss = data_loss

    def learn(self, epoches, solution, return_output=False, debug_mode=False):
        lowest_loss = self.loss
        for i in range(len(self.hidden_layers)):
            exec(f"self.best_weights{i} = self.weights{i}.copy()")
            exec(f"self.best_biases{i} = self.biases{i}.copy()")
        for cycle in range(epoches):
            for a in range(len(self.hidden_layers)):
                if a == 0:
                    exec(f"self.weights{a} += 0.05 * np.random.randn(self.inputs, self.hidden_layers[0])")
                    exec(f"self.biases{a} += 0.05 * np.random.randn(1, self.hidden_layers[0])")
                else:
                    try:
                        exec(f"self.weights{a} += 0.05 * np.random.randn(self.hidden_layers[{a}], self.hidden_layers[{a} + 1])")
                        exec(f"self.biases{a} = 0.05 * np.random.randn(1, self.hidden_layers[{a}])")
                    except Exception:
                        exec(f"self.weights{a} += 0.05 * np.random.randn(self.hidden_layers[{a}], self.outputs)")
                        exec(f"self.biases{a} += 0.05 * np.random.randn(1, self.hidden_layers[{a}])")

            self.feed_forward()
            
            self.back_propagation(solution)
            
            self.predictions = np.argmax(self.output, axis=1)
            self.accuracy = np.mean(self.predictions[0]==solution)
            
            if self.loss < lowest_loss:
                if debug_mode == True:
                    print(f"Better set of weights found, replacing. Loss: {self.loss} Accuracy: {self.accuracy}")
                for i in range(len(self.hidden_layers)):
                    exec(f"self.best_weights{i} = self.weights{i}.copy()")
                    exec(f"self.best_biases{i} = self.biases{i}.copy()")
                lowest_loss = self.loss
            
            else:
                for i in range(len(self.hidden_layers)):
                    exec(f"self.weights{i} = self.best_weights{i}.copy()")
                    exec(f"self.biases{i} = self.best_biases{i}.copy()")

    def set_function(self, index, function):
        if index <= len(self.hidden_layers):
            exec(f"self.function{index} = '{function}'")
        else:
            raise Exception
    
    def set_weights_and_biases(self, index, weights=None, biases=None):
        if weights is None and biases is None:
            if index == 0:
                exec(f"self.weights{index} = 0.01 * np.random.randn(self.inputs, self.hidden_layers[0])")
                exec(f"self.biases{index} = np.zeros((1, self.hidden_layers[0]))")
            else:
                try:
                    exec(f"self.weights{index} = 0.01 * np.random.randn(self.hidden_layers[{index}], self.hidden_layers[{index} + 1])")
                    exec(f"self.biases{index} = np.zeros((1, self.hidden_layers[{index}]))")
                except Exception:
                    exec(f"self.weights{index} = 0.01 * np.random.randn(self.hidden_layers[{index}], self.outputs)")
                    exec(f"self.biases{index} = np.zeros((1, self.hidden_layers[{index}]))")
        else:
            exec(f"self.weights{index} = {weights}")
            exec(f"self.biases{index} = {biases}")
    
    def check(self):
        for i in range(len(self.hidden_layers)):
            exec(f"if 'function{i}' not in self.__dict__: raise Exception")
        for i in range(len(self.hidden_layers)):
            exec(f"if 'weights{i}' not in self.__dict__: raise Exception")
        for i in range(len(self.hidden_layers)):
            exec(f"if 'biases{i}' not in self.__dict__: raise Exception")
        return True
    
    def quick_setup(self, input_data):
        self.input_data = input_data
        for i in range(len(self.hidden_layers)):
            self.set_function(i, None)
        for i in range(len(self.hidden_layers)):
            self.set_weights_and_biases(i)
        self.check()
