import random


def initialize_weight():
    return random.uniform(-0.5, 0.5)

def exp(x, terms=20):
    result = 1.0
    term = 1.0
    for n in range(1, terms):
        term *= x / n
        result += term
    return result


def tanh(x):
    if x > 100:  # Prevent overflow
        return 1.0
    elif x < -100:  # Prevent underflow
        return -1.0
    e_2x = exp(2 * x)  
    return (e_2x - 1) / (e_2x + 1)

def tanh_derivative(x):
    return 1 - tanh(x) ** 2


def forward_pass(inputs, weights, biases):
    h1 = tanh(inputs[0] * weights[0] + inputs[1] * weights[1] + biases[0])
    h2 = tanh(inputs[0] * weights[2] + inputs[1] * weights[3] + biases[0])
    
    o1 = tanh(h1 * weights[4] + h2 * weights[5] + biases[1])
    o2 = tanh(h1 * weights[6] + h2 * weights[7] + biases[1])
    
    return o1, o2, h1, h2

def mse_loss(output, target):
    return 0.5 * ((output[0] - target[0]) ** 2 + (output[1] - target[1]) ** 2)

def backpropagation(inputs, weights, biases, output, target, learning_rate=0.1):
    o1, o2, h1, h2 = output
    
   
    d_o1 = (o1 - target[0]) * tanh_derivative(h1 * weights[4] + h2 * weights[5] + biases[1])
    d_o2 = (o2 - target[1]) * tanh_derivative(h1 * weights[6] + h2 * weights[7] + biases[1])
    
    
    d_h1 = (d_o1 * weights[4] + d_o2 * weights[6]) * tanh_derivative(inputs[0] * weights[0] + inputs[1] * weights[1] + biases[0])
    d_h2 = (d_o1 * weights[5] + d_o2 * weights[7]) * tanh_derivative(inputs[0] * weights[2] + inputs[1] * weights[3] + biases[0])
    
   
    weights[4] -= learning_rate * d_o1 * h1
    weights[5] -= learning_rate * d_o1 * h2
    weights[6] -= learning_rate * d_o2 * h1
    weights[7] -= learning_rate * d_o2 * h2
    
   
    weights[0] -= learning_rate * d_h1 * inputs[0]
    weights[1] -= learning_rate * d_h1 * inputs[1]
    weights[2] -= learning_rate * d_h2 * inputs[0]
    weights[3] -= learning_rate * d_h2 * inputs[1]
    
    
    biases[0] -= learning_rate * (d_h1 + d_h2)
    biases[1] -= learning_rate * (d_o1 + d_o2)
    
    return weights, biases


inputs = [0.8, 0.6]


weights = [initialize_weight() for _ in range(8)]
biases = [0.5, 0.7]
target = [0.5, 0.7]


output = forward_pass(inputs, weights, biases)
print("Initial output of the network:", (output[0], output[1]))

weights, biases = backpropagation(inputs, weights, biases, output, target)

print("Updated weights:", weights)
print("Updated biases:", biases)

updated_output = forward_pass(inputs, weights, biases)
print("Updated output of the network:", (updated_output[0], updated_output[1]))