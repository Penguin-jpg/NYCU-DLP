import numpy as np
import matplotlib.pyplot as plt


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
    return 1.0 / (1.0 + np.exp(-x))


# derivation from https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
def derivative_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return np.where(x > 0, 1, 0)


# loss function (y: ground-truth, y_hat: prediction) and derivative
# because this is a classification task, we use binary cross-entropy
def binary_cross_entropy(y, y_hat):
    # calculate like this because input one data point at a time
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def derivative_binary_cross_entropy(y, y_hat):
    # dL/d(y_hat) = -(y/y_hat - (1-y)/(1-y_hat)) = -(y-y_hat)/(y_hat * (1-y_hat))
    return -(y / y_hat - (1 - y) / (1 - y_hat))


class Network:
    def __init__(self, input_dim=2, hidden_dims=[4, 4, 4], output_dim=1, activation="sigmoid", lr=1e-2):
        # every layer contains its own weight and bias
        # shape of weight: (out_dim, in_dim) initialize as random
        # shape of bias: (out_dim, 1) initialize as 0

        # input layer
        self.w1 = np.random.randn(hidden_dims[0], input_dim)  # shape: (4, 2)
        self.b1 = np.zeros((hidden_dims[0], 1))

        # hidden layer 1
        self.w2 = np.random.randn(hidden_dims[1], hidden_dims[0])  # shape: (4, 4)
        self.b2 = np.zeros((hidden_dims[1], 1))

        # hidden layer 2
        self.w3 = np.random.randn(hidden_dims[2], hidden_dims[1])  # shape: (4, 4)
        self.b3 = np.zeros((hidden_dims[2], 1))

        # output layer
        self.w4 = np.random.randn(output_dim, hidden_dims[2])  # shape: (1, 4)
        self.b4 = np.zeros((output_dim, 1))

        # activation function
        if activation == "sigmoid":
            self.act = sigmoid
            self.derivative_act = derivative_sigmoid
        elif activation == "relu":
            self.act = relu
            self.derivative_act = derivative_relu

        # learning rate
        self.lr = lr

        # for backpropagation, we need to store computed values
        self.values = {}

    def forward(self, x):
        # shape of input: (2, 1) one data point at a time
        self.values["x"] = x

        # for each layer, calculate a=act(z=w^Tx+b)
        # z1 = np.dot(self.w1, x) + self.b1  # shape: (4, 1)
        # a1 = self.act(z1)
        # self.values["z1"] = z1
        # self.values["a1"] = a1

        # z2 = np.dot(self.w2, a1) + self.b2  # shape: (4, 1)
        # a2 = self.act(z2)
        # self.values["z2"] = z2
        # self.values["a2"] = a2

        # z3 = np.dot(self.w3, a2) + self.b3  # shape: (4, 1)
        # a3 = self.act(z3)
        # self.values["z3"] = z3
        # self.values["a3"] = a3

        # z4 = np.dot(self.w4, a3) + self.b4  # shape: (1, 1)
        # y = self.act(z4)
        # self.values["z4"] = z4
        # self.values["a4"] = y

        z1 = np.dot(self.w1, x)  # shape: (4, 1)
        a1 = self.act(z1)
        self.values["z1"] = z1
        self.values["a1"] = a1

        z2 = np.dot(self.w2, a1)  # shape: (4, 1)
        a2 = self.act(z2)
        self.values["z2"] = z2
        self.values["a2"] = a2

        z3 = np.dot(self.w3, a2)  # shape: (4, 1)
        a3 = self.act(z3)
        self.values["z3"] = z3
        self.values["a3"] = a3

        z4 = np.dot(self.w4, a3)  # shape: (1, 1)
        y = self.act(z4)
        self.values["z4"] = z4
        self.values["a4"] = y

        return y

    def backward(self, y, y_hat):
        # backprogagation starts from output layer (using chain rule)

        # the graident of output layer:
        # dL/d(w4) = dL/d(y_hat) * d(y_hat)/d(z4) * d(z4)/d(w4)
        # dL/d(b4) = dL/d(y_hat) * d(y_hat)/d(z4) * d(z4)/d(b4)
        d_L_d_y_hat = derivative_binary_cross_entropy(y, y_hat)  # y_hat = a4, shape: (1, 1)
        d_y_hat_d_z4 = self.derivative_act(self.values["z4"])  # d(a4)/d(z4)=act'(z4), shape: (1, 1)
        d_z4_d_w4 = self.values["a3"]  # d(z4)/d(w4)=a3, shape: (4, 1)
        d_L_d_z4 = np.dot(d_L_d_y_hat, d_y_hat_d_z4)  # shape: (1, 1)
        d_L_d_w4 = np.dot(d_L_d_z4, d_z4_d_w4.T)  # shape: (1, 1) dot (1, 4) = (1, 4)
        # d_z4_d_b4 = 1  # d(z4)/d(b4)=1, shape: (1)
        # d_L_d_b4 = d_L_d_z4 * d_z4_d_b4  # shape: (1, 1)

        # the graident of hidden layer 2:
        # dL/d(w3) = dL/d(z4) * d(z4)/d(a3) * d(a3)/d(z3) * d(z3)/d(w3)
        # dL/d(b3) = dL/d(z4) * d(z4)/d(a3) * d(a3)/d(z3) * d(z3)/d(b3)
        d_z4_d_a3 = self.w4  # d(z4)/d(a3)=w4  (a3 is the "x"), shape: (1, 4)
        d_a3_d_z3 = self.derivative_act(self.values["z3"])  # d(a3)/d(z3)=act'(z3), shape: (4, 1)
        d_z3_d_w3 = self.values["a2"]  # d(z3)/d(w3)=a2, shape: (4, 1)
        d_L_d_z3 = np.dot(np.dot(d_L_d_z4, d_z4_d_a3), d_a3_d_z3)  # shape: (1, 1) dot (1, 4) dot (4, 1) = (1, 1)
        d_L_d_w3 = np.dot(d_L_d_z3, d_z3_d_w3.T)  # shape: (1, 1) dot (1, 4) = (1, 4)
        # d_z3_d_b3 = 1  # d(z3)/d(b3)=1, shape: (1)
        # d_L_d_b3 = d_L_d_z3 * d_z3_d_b3  # shape: (1, 1)

        # the graident of hidden layer 1:
        # dL/d(w2) = dL/d(z3) * d(z3)/d(a2) * d(a2)/d(z2) * d(z2)/d(w2)
        # dL/d(b2) = dL/d(z3) * d(z3)/d(a2) * d(a2)/d(z2) * d(z2)/d(b2)
        d_z3_d_a2 = self.w3  # d(z3)/d(a2)=w3 (a2 is the "x"), shape: (4, 4)
        d_a2_d_z2 = self.derivative_act(self.values["z2"])  # d(a2)/d(z2)=act'(z2), shape: (4, 1)
        d_z2_d_w2 = self.values["a1"]  # d(z2)/d(w2)=a1, shape: (4, 1)
        d_L_d_z2 = np.dot(d_L_d_z3 * d_z3_d_a2, d_a2_d_z2)  # shape: (1, 1) * (4, 4) dot (4, 1) = (4, 1)
        d_L_d_w2 = np.dot(d_L_d_z2, d_z2_d_w2.T)  # shape: (1, 1) dot (1, 4) = (1, 4)
        # d_z2_d_b2 = 1  # d(z3)/d(b3)=1, shape: (1)
        # d_L_d_b2 = d_L_d_z2 * d_z2_d_b2  # shape: (1, 4)

        # the graident of input layer:
        # dL/d(w1) = dL/d(z2) * d(z2)/d(a1) * d(a1)/d(z1) * d(z1)/d(w1)
        # dL/d(b1) = dL/d(z2) * d(z2)/d(a1) * d(a1)/d(z1) * d(z1)/d(b1)
        d_z2_d_a1 = self.w2  # d(z2)/d(a1)=w2 (a1 is the "x"), shape: (4, 4)
        d_a1_d_z1 = self.derivative_act(self.values["z1"])  # d(a1)/d(z1)=act'(z1), shape: (4, 1)
        d_z1_d_w1 = self.values["x"]  # d(z1)/d(w1)=x, shape: (2, 1)
        d_L_d_z1 = np.dot(np.dot(d_L_d_z2.T, d_z2_d_a1), d_a1_d_z1)  # shape: (1, 4) dot (4, 4) dot (4, 1) = (1, 1)
        d_L_d_w1 = np.dot(d_L_d_z1, d_z1_d_w1.T)  # shape: (1, 1)dot (1, 2) = (1, 2)
        # d_z1_d_b1 = 1  # d(z1)/d(b1)=1, shape: (1)
        # d_L_d_b1 = d_L_d_z1 * d_z1_d_b1  # shape: (1, 2)

        # update weights and biases
        self.w1 -= self.lr * d_L_d_w1
        # self.b1 -= self.lr * d_L_d_b1
        self.w2 -= self.lr * d_L_d_w2
        # self.b2 -= self.lr * d_L_d_b2
        self.w3 -= self.lr * d_L_d_w3
        # self.b3 -= self.lr * d_L_d_b3
        self.w4 -= self.lr * d_L_d_w4
        # self.b4 -= self.lr * d_L_d_b4


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


def train_loop(model, inputs, labels, num_epochs):
    for epoch in range(num_epochs):
        loss = 0
        for input, label in zip(inputs, labels):
            # input shape needs to be (2, 1) so that it can be multiplied with weight
            # make shape of input (2, 1)
            input = input.reshape(2, 1)

            # feed to network
            y_hat = model.forward(input)

            # calculate loss
            loss += binary_cross_entropy(label, y_hat)

            model.backward(label, y_hat)

        loss /= len(inputs)

        if epoch % 5000 == 0:
            print(f"epoch {epoch} loss : {loss[0][0]}")


def test(model, inputs, labels):
    accuracy = 0
    loss = 0
    results = []
    predictions = []

    for i, (input, label) in enumerate(zip(inputs, labels)):
        input = input.reshape(2, 1)
        y_hat = model.forward(input)
        results.append(y_hat[0][0])
        predictions.append(np.round(y_hat))
        loss += binary_cross_entropy(label, y_hat)
        accuracy += 1 if np.round(y_hat) == label else 0
        print(f"Iter{i} |   Ground truth: {label[0]:.1f} |   Prediction: {y_hat[0][0]:.5f} |")

    print(results)

    loss /= len(inputs)
    accuracy /= len(inputs)
    print(f"loss={loss[0][0]:.5f} accuracy={accuracy * 100:.2f}%")

    show_results(inputs, labels, predictions)


if __name__ == "__main__":
    # constants
    DATASET_SIZE = 100
    NUM_EPOCHS = 500
    ACTIVATION = "sigmoid"
    LR = 1e-3

    inputs, labels = generate_linear(DATASET_SIZE)
    # inputs, labels = generate_XOR_easy()

    # create network
    net = Network(input_dim=2, hidden_dims=[4, 4, 4], output_dim=1, activation=ACTIVATION, lr=LR)

    # train network
    train_loop(net, inputs, labels, NUM_EPOCHS)
    test(net, inputs, labels)
