#katherine duque's PLA
import numpy as np #structures
import matplotlib.pyplot as plt #visualization

#first, we need to create a random line in the plane within the set boundaries.

def create_line():
    m = np.random.uniform(-1, 1)
    c = np.random.uniform(-1, 1)

    def line_function(x):
        return m * x + c
    return line_function


#now we need to generate datapoints on either side of the line
#the line serves as a boundary that divided the set into its
#respective categories which the perceptron will then
#attempt to sort.

def generate_data_points(line_func, num_points=100):
    data = np.random.uniform(-1, 1, (num_points, 2))
    #single line notation looks so nice :)
    labels = np.array([1 if y > line_func(x) else -1 for x, y in data])
    return data, labels

#i will be implementing the perceptron below using a class.
#encapsulation of the perceptron's behavior makes this look neater
#this also just makes the code easier to read :)

class Perceptron:
    def __init__(self):
        self.weights = np.random.uniform(-1, 1, 3) #3 = 2 dimensions of the data set+ a bias term

#the +1 or -1 dot product predicts which class a feature belongs to. 

    def predict(self, inputs):
        return np.sign(np.dot(inputs, self.weights))

#adjust the perceptron weights based on correct classification
#thereby allowing it to "learn" from its past mistakes.
#keep count of how many times the weights were adjusted

    def train(self, training_inputs, labels):
        count = 0
        while True:
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if prediction != label:
                    self.weights += label * inputs
                    errors += 1
                    count += 1
            if errors == 0:
                break
        return count

#here we want to create the objects

line_func = create_line()
data, labels = generate_data_points(line_func)
perceptron = Perceptron()

#we create a 2D awway where each element's length is equal to the number of data
#points and the column represents the bias term for each generated data point. 
#after augmenting the arrays appended with 'np.hstack', the features and labels are added
#to 'labelled_data'. 

bias = np.ones((data.shape[0], 1))
training_data = np.hstack((data, bias))
labelled_data = np.column_stack((training_data, labels))

#this trains the cute little PLA. copying the weight insures that its original value
#will not be modified during training. then, when iterating with the PLA it keeps
#track of how many data points it has misclassified. the number of iterations the
#percptron has taken to learn to sort the dataset is stored in 'update_count'. 

initial_weights = perceptron.weights.copy()
update_count = perceptron.train(training_data, labels)
final_weights = perceptron.weights

#here we plot the data

def plot_perceptron(data, line_func, weights, title):

    plt.figure(figsize=(10, 8))
    plt.scatter(data[:,0], data[:,1], c=labels, cmap='Spectral') #the built in colors suck and are kinda depressing
    x_vals = np.linspace(-1, 1, 100) #here you can change the value to 1000 for part d). 
    plt.plot(x_vals, line_func(x_vals), label="Original Line", color='red')
    plt.plot(x_vals, -(weights[2] + weights[0]*x_vals) / weights[1], label="Perceptron Boundary", color='green')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(title)
    plt.legend()
    plt.show()

#plot the first iteration
plot_perceptron(data, line_func, initial_weights, "Perceptron's first guess")

#plot the sorted data 
plot_perceptron(data, line_func, final_weights, "Perceptron's educated guess! so smart")

#show the iterations
print(f"How many times did it take to converge?: {update_count}")

#of course, we will find that the number of times the weights must be updated increases dramatically. 
