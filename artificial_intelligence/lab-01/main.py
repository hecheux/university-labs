from numpy import exp, array, random, dot
import time


class NeuralNetwork():
    def __init__(self, seed_value=2):
        random.seed(seed_value)

        # 5 весов для 5 входов
        self.synaptic_weights = 2 * random.random((5, 1)) - 1

        # История ошибки
        self.mse_history = []

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # В train добавили параметр eps
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, eps=0.001):
        self.mse_history = []

        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output

            mse = (error ** 2).mean()
            self.mse_history.append(mse)

            if mse < eps:
                return iteration + 1   # количество реально выполненных шагов

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

        return number_of_training_iterations

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def get_weights(self):
        return self.synaptic_weights.flatten()


if __name__ == "__main__":
    training_set_inputs = array([
        [0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1],

        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
    ])

    training_set_outputs = array([
        [0, 1, 1, 0, 0, 1, 1, 0]
    ]).T


    thresholds = [0.1, 0.05, 0.01, 0.005, 0.001]
    seed = 20

    for eps in thresholds:
        neural_network = NeuralNetwork(seed)

        start_time = time.perf_counter()

        steps = neural_network.train(
            training_set_inputs,
            training_set_outputs,
            100000,
            eps
        )

        end_time = time.perf_counter()
        training_time = end_time - start_time

        print(f"{eps:<5} | {steps} | {training_time:.6f} сек")