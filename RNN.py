import random

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Word mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # Initialize weights
        self.Wxh = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(hidden_size)]
        self.Whh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Why = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(vocab_size)]
        self.bh = [0.0 for _ in range(hidden_size)]
        self.by = [0.0 for _ in range(vocab_size)]
        
        # Training cache
        self.h_prev = None
    
    def build_vocab(self, text):
        words = text.split()
        unique_words = sorted(list(set(words)))
        self.word_to_idx = {w: i for i, w in enumerate(unique_words)}
        self.idx_to_word = {i: w for i, w in enumerate(unique_words)}
        self.vocab_size = len(unique_words)
        
        # Reinitialize weights
        self.Wxh = [[random.uniform(-0.1, 0.1) for _ in range(self.vocab_size)] for _ in range(self.hidden_size)]
        self.Whh = [[random.uniform(-0.1, 0.1) for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
        self.Why = [[random.uniform(-0.1, 0.1) for _ in range(self.hidden_size)] for _ in range(self.vocab_size)]
    
    def one_hot_encode(self, word):
        vec = [0.0] * self.vocab_size
        vec[self.word_to_idx[word]] = 1.0
        return vec
    
    def forward(self, inputs):
        # Initialize hidden state and storage
        self.h_prev = [0.0] * self.hidden_size
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = self.h_prev.copy()
        
        # Forward pass through each time step
        for t in range(len(inputs)):
            xs[t] = self.one_hot_encode(inputs[t])
            
            # Calculate new hidden state
            h = [0.0] * self.hidden_size
            for i in range(self.hidden_size):
                # Wxh * x
                for j in range(self.vocab_size):
                    h[i] += self.Wxh[i][j] * xs[t][j]
                # Whh * h_prev
                for j in range(self.hidden_size):
                    h[i] += self.Whh[i][j] * hs[t-1][j]
                # Add bias and apply tanh
                h[i] = self.tanh(h[i] + self.bh[i])
            hs[t] = h
            
            # Calculate output
            y = [0.0] * self.vocab_size
            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    y[i] += self.Why[i][j] * h[j]
                y[i] += self.by[i]
            ys[t] = y
            
            # Softmax
            ps[t] = self.softmax(y)
        
        return xs, hs, ps
    
    def backward(self, xs, hs, ps, target_word):
        # Initialize gradients
        dWxh = [[0.0 for _ in range(self.vocab_size)] for _ in range(self.hidden_size)]
        dWhh = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
        dWhy = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.vocab_size)]
        dbh = [0.0 for _ in range(self.hidden_size)]
        dby = [0.0 for _ in range(self.vocab_size)]
        dh_next = [0.0 for _ in range(self.hidden_size)]
        
        target_idx = self.word_to_idx[target_word]
        
        # Backward pass through time
        for t in reversed(range(len(xs))):
            # Output gradient
            dy = list(ps[t])
            dy[target_idx] -= 1.0
            
            # dWhy and dby
            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    dWhy[i][j] += dy[i] * hs[t][j]
                dby[i] += dy[i]
            
            # Hidden gradient
            dh = [0.0 for _ in range(self.hidden_size)]
            for j in range(self.hidden_size):
                # Backprop into h
                dh[j] = 0.0
                for i in range(self.vocab_size):
                    dh[j] += self.Why[i][j] * dy[i]
                dh[j] += dh_next[j]
                
                # Backprop through tanh
                dtanh = (1 - hs[t][j] * hs[t][j]) * dh[j]
                
                # dbh
                dbh[j] += dtanh
                
                # dWxh and dWhh
                for k in range(self.vocab_size):
                    dWxh[j][k] += dtanh * xs[t][k]
                for k in range(self.hidden_size):
                    dWhh[j][k] += dtanh * hs[t-1][k]
                
                # Store for next step
                dh_next[j] = 0.0
                for k in range(self.hidden_size):
                    dh_next[j] += self.Whh[k][j] * dtanh
        
        # Clip gradients
        self.clip_gradients(dWxh)
        self.clip_gradients(dWhh)
        self.clip_gradients(dWhy)
        self.clip_gradients([dbh])
        self.clip_gradients([dby])
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def update_parameters(self, grads):
        dWxh, dWhh, dWhy, dbh, dby = grads
        
        # Update weights with learning rate
        for i in range(self.hidden_size):
            for j in range(self.vocab_size):
                self.Wxh[i][j] -= self.learning_rate * dWxh[i][j]
            for j in range(self.hidden_size):
                self.Whh[i][j] -= self.learning_rate * dWhh[i][j]
        
        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                self.Why[i][j] -= self.learning_rate * dWhy[i][j]
        
        # Update biases
        for i in range(self.hidden_size):
            self.bh[i] -= self.learning_rate * dbh[i]
        for i in range(self.vocab_size):
            self.by[i] -= self.learning_rate * dby[i]
    
    def train(self, input_sequence, target_word, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            xs, hs, ps = self.forward(input_sequence)
            
            # Backward pass
            grads = self.backward(xs, hs, ps, target_word)
            
            # Update parameters
            self.update_parameters(grads)
            
            # Calculate loss
            loss = 0.0
            for t in ps:
                p = ps[t][self.word_to_idx[target_word]]
                loss += -self.log(p) if p > 0 else 20  # Prevent log(0)
            loss /= len(ps)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, input_sequence):
        _, _, ps = self.forward(input_sequence)
        last_p = ps[len(input_sequence)-1]
        max_idx = max(range(len(last_p)), key=lambda i: last_p[i])
        return self.idx_to_word[max_idx]
    
    # Helper functions
    def tanh(self, x):
        return (self.exp(x) - self.exp(-x)) / (self.exp(x) + self.exp(-x) + 1e-8)
    
    def exp(self, x):
        # Simple approximation
        return 1 + x + (x*x)/2 + (x*x*x)/6
    
    def softmax(self, x):
        max_x = max(x)
        exp_x = [self.exp(i - max_x) for i in x]  # Numerical stability
        sum_exp = sum(exp_x)
        return [i / (sum_exp + 1e-8) for i in exp_x]
    
    def log(self, x):
        # Simple approximation
        return (x - 1) - (x - 1)**2/2 + (x - 1)**3/3 if x > 0 else -20
    
    def clip_gradients(self, grad, max_val=5.0):
        for i in range(len(grad)):
            if isinstance(grad[i], list):
                for j in range(len(grad[i])):
                    if grad[i][j] > max_val:
                        grad[i][j] = max_val
                    elif grad[i][j] < -max_val:
                        grad[i][j] = -max_val
            else:
                if grad[i] > max_val:
                    grad[i] = max_val
                elif grad[i] < -max_val:
                    grad[i] = -max_val

# Example usage
if __name__ == "__main__":
    # Sample training data
    text = "I love python programming"
    words = text.split()
    
    # Create RNN
    rnn = SimpleRNN(vocab_size=4, hidden_size=5, learning_rate=0.1)
    rnn.build_vocab(text)
    
    # Training data
    input_sequence = words[:3]
    target_word = words[3]
    
    print("Before training:")
    print(f"Input: {' '.join(input_sequence)}")
    print(f"Prediction: {rnn.predict(input_sequence)}")
    print(f"Target: {target_word}")
    
    # Train
    rnn.train(input_sequence, target_word, epochs=100)
    
    print("\nAfter training:")
    print(f"Input: {' '.join(input_sequence)}")
    print(f"Prediction: {rnn.predict(input_sequence)}")
    print(f"Target: {target_word}")