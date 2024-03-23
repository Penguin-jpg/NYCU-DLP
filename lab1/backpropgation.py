import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


# dataset generator function
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        # distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs, labels = [], []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1.0 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


# activation functions and derivatives
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x + 1e-8))


# derivation from https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
def derivative_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return np.where(x > 0, 1, 0)


# loss function (y: ground-truth, y_hat: prediction) and derivative
def binary_cross_entropy(y, y_hat):
    # calculate like this because input one data point at a time
    return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))


def derivative_binary_cross_entropy(y, y_hat):
    # dC/d(y_hat) = -(y/y_hat - (1-y)/(1-y_hat)) = -(y-y_hat)/(y_hat * (1-y_hat))
    return -(y / (y_hat + 1e-8) - (1 - y) / (1 - y_hat + 1e-8))


def sum_of_square_erorr(y, y_hat):
    return 0.5 * np.sum((y - y_hat) ** 2)


def derivative_sum_of_square_erorr(y, y_hat):
    # dC/d(y_hat) = 2 * 1/2 * (y - y_hat)^{2-1} * -1 = y_hat - y
    return y_hat - y


def show_results(x, y, y_hat):
    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize=18)
    for i in range(len(x)):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.subplot(1, 2, 2)
    plt.title("Predict result", fontsize=18)
    for i in range(len(x)):
        if y_hat[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.show()


def show_training_plot(num_epochs, losses):
    plt.title("Trainging loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(num_epochs), losses)
    plt.show()


class Network:
    def __init__(self, input_dim=2, hidden_dims=[4, 4], output_dim=1, lr=1e-2):
        # every layer contains its own weight and bias
        # shape of weight: (out_dim, in_dim) initialize as random
        # shape of bias: (out_dim, 1) initialize as 0

        # input layer
        self.w1 = np.random.randn(hidden_dims[0], input_dim)  # shape: (4, 2)
        self.b1 = np.zeros((hidden_dims[0], 1))

        # hidden layer 1
        self.w2 = np.random.randn(hidden_dims[0], hidden_dims[0])  # shape: (4, 4)
        self.b2 = np.zeros((hidden_dims[0], 1))

        # hidden layer 2
        self.w3 = np.random.randn(hidden_dims[1], hidden_dims[0])  # shape: (4, 4)
        self.b3 = np.zeros((hidden_dims[1], 1))

        # output layer
        self.w4 = np.random.randn(output_dim, hidden_dims[1])  # shape: (1, 4)
        self.b4 = np.zeros((output_dim, 1))

        # learning rate
        self.lr = lr

        # for backpropagation, we need to store computed values
        self.values = {}

    def forward(self, x):
        # shape of input: (2, N)
        self.values["x"] = x

        # for each layer, calculate a=act(z=w^Tx+b)
        z1 = np.dot(self.w1, x) + self.b1  # shape: (4, N)
        a1 = relu(z1)
        self.values["z1"] = z1
        self.values["a1"] = a1

        z2 = np.dot(self.w2, a1) + self.b2  # shape: (4, N)
        a2 = relu(z2)
        self.values["z2"] = z2
        self.values["a2"] = a2

        z3 = np.dot(self.w3, a2) + self.b3  # shape: (4, N)
        a3 = sigmoid(z3)
        self.values["z3"] = z3
        self.values["a3"] = a3

        z4 = np.dot(self.w4, a3) + self.b4  # shape: (1, N)
        y = sigmoid(z4)
        self.values["z4"] = z4
        self.values["a4"] = y

        return y

    def backward(self, y, y_hat):
        # to perform backpropagation, we need to use four formulas, I do the proof in the report
        # 1. delta^L = dC/d(y_hat) * derivative_act(z^L)
        #    - delta: error, C: loss (cost), L: the last layer, *: element-wise multiplication
        # 2. delta^l = ((w^{l+1})^T dot delta^{l+1}) * derivative_act(z^l)
        #    - dot: dot product (or matrix multiplication)
        # 3. dC/d(w^l) = a^{l-1} dot delta^l (graident for weight)
        # 4. dC/d(b^l) = delta^l (graident for bias)

        # batch size
        N = y.shape[1]

        # graident of output layer
        d_C_d_y_hat = derivative_binary_cross_entropy(y, y_hat)
        # apply formula 1
        delta4 = d_C_d_y_hat * derivative_sigmoid(self.values["z4"])
        # apply formula 3
        # shape: (1, N) dot (4, N)^T = (1, 4)
        d_C_d_w4 = np.dot(d_C_d_y_hat, self.values["a3"].T) / N
        # apply formula 4
        d_C_d_b4 = np.sum(delta4, axis=1, keepdims=True) / N

        # graident of hidden layer 2
        # apply formula 2
        # shape: (1, 4)^T dot (1, N) = (4, N)
        delta3 = np.dot(self.w4.T, delta4) * derivative_sigmoid(self.values["z3"])
        # apply formula 3 and 4
        d_C_d_w3 = np.dot(delta3, self.values["a2"].T) / N  # shape: (4, N) dot (4, N)^T = (4, 4)
        d_C_d_b3 = np.sum(delta3, axis=1, keepdims=True) / N

        # graident of hidden layer 1
        # apply formula 2
        # shape: (4, 4)^T dot (4, N) = (4, N)
        delta2 = np.dot(self.w3.T, delta3) * derivative_relu(self.values["z2"])
        # apply formula 3 and 4
        d_C_d_w2 = np.dot(delta2, self.values["a1"].T) / N  # shape: (4, N) dot (4, N)^T = (4, 4)
        d_C_d_b2 = np.sum(delta2, axis=1, keepdims=True) / N

        # gradient of input layer
        # apply formula 2
        # shape: (4, 4)^T dot (4, N) = (4, N)
        delta1 = np.dot(self.w2.T, delta2) * derivative_relu(self.values["z1"])
        # apply formula 3 and 4
        d_C_d_w1 = np.dot(delta1, self.values["x"].T) / N  # shape: (4, N) dot (2, N)^T = (4, 2)
        d_C_d_b1 = np.sum(delta1, axis=1, keepdims=True) / N

        # update weights and biases
        self.w1 -= self.lr * d_C_d_w1
        self.b1 -= self.lr * d_C_d_b1
        self.w2 -= self.lr * d_C_d_w2
        self.b2 -= self.lr * d_C_d_b2
        self.w3 -= self.lr * d_C_d_w3
        self.b3 -= self.lr * d_C_d_b3
        self.w4 -= self.lr * d_C_d_w4
        self.b4 -= self.lr * d_C_d_b4

    def train(self, inputs, labels, num_epochs):
        losses = []

        for epoch in range(num_epochs):
            loss = 0

            # reshape input to (N, 2) and transpose so that it can be multiplied with weight
            inputs = inputs.reshape(-1, 2).T
            # prediction
            y_hat = self.forward(inputs)
            # calculate loss
            loss = binary_cross_entropy(labels.T, y_hat)
            # backpropagation
            self.backward(labels.T, y_hat)

            losses.append(loss)

            if epoch % 5000 == 0 or epoch == num_epochs - 1:
                print(f"epoch {epoch} loss : {loss}")

        show_training_plot(num_epochs, losses)

    def train_SGD(self, inputs, labels, num_epochs, batch_size):
        rng = np.random.default_rng(seed=42)
        losses = []

        for epoch in range(num_epochs):
            loss = 0

            # for every epoch, randomly shuffle the dataset
            random_indices = rng.choice(len(inputs), size=len(inputs), replace=False)
            # split mini batches
            batches = np.array_split(random_indices, len(inputs) // batch_size)

            for batch in batches:
                # assemble mini batch
                batched_x = inputs[batch]
                batched_y = labels[batch]

                batched_x = batched_x.reshape(-1, 2).T

                y_hat = self.forward(batched_x)

                loss += binary_cross_entropy(batched_y.T, y_hat)

                self.backward(batched_y.T, y_hat)

            # average over the number of batches
            loss /= len(batches)
            losses.append(loss)

            if epoch % 5000 == 0 or epoch == num_epochs - 1:
                print(f"epoch {epoch} loss : {loss}")

        show_training_plot(num_epochs, losses)

    def test(self, inputs, labels):
        accuracy = 0
        loss = 0
        results = []
        predictions = []

        for i, (input, label) in enumerate(zip(inputs, labels)):
            input = input.reshape(2, 1)
            y_hat = self.forward(input)
            results.append(y_hat[0][0])
            predictions.append(np.round(y_hat))
            loss += binary_cross_entropy(label, y_hat)
            accuracy += 1 if np.round(y_hat) == label else 0
            print(f"Iter{i} |   Ground truth: {label[0]:.1f} |   Prediction: {y_hat[0][0]:.5f} |")

        loss /= len(inputs)
        accuracy /= len(inputs)
        print(f"loss={loss:.5f} accuracy={accuracy * 100:.2f}%")
        print("test results:")
        print(np.array([results]))

        show_results(inputs, labels, predictions)

    def save(self, model_path):
        import os

        np.save(os.path.join(model_path, "w1.npy"), self.w1)
        np.save(os.path.join(model_path, "w2.npy"), self.w2)
        np.save(os.path.join(model_path, "w3.npy"), self.w3)
        np.save(os.path.join(model_path, "w4.npy"), self.w4)

    def load(self, model_path):
        import os

        self.w1 = np.load(os.path.join(model_path, "w1.npy"))
        self.w2 = np.load(os.path.join(model_path, "w2.npy"))
        self.w3 = np.load(os.path.join(model_path, "w3.npy"))
        self.w4 = np.load(os.path.join(model_path, "w4.npy"))


if __name__ == "__main__":
    # train network (linear dataset)
    # net = Network(input_dim=2, hidden_dims=[4, 4], output_dim=1, lr=1e-4)
    net = Network(input_dim=2, hidden_dims=[8, 8], output_dim=1, lr=1e-4)
    inputs, labels = generate_linear()
    # net.train(inputs, labels, 30000)
    net.train_SGD(inputs, labels, 30000, 10)
    net.test(inputs, labels)
    # net.load("weights/linear")
    # net.save("weights/linear")

    # xor dataset
    # net = Network(input_dim=2, hidden_dims=[4, 4], output_dim=1, lr=1e-3)
    net = Network(input_dim=2, hidden_dims=[8, 8], output_dim=1, lr=1e-3)
    inputs, labels = generate_XOR_easy()
    # net.train(inputs, labels, 20000)
    net.train_SGD(inputs, labels, 20000, 7)
    net.test(inputs, labels)
    # net.save("weights/xor")
    # net.load("weights/xor")
