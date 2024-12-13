from neat.nn import transformer
import neat
from visualize import draw_net, plot_stats, plot_species
import pickle
import pandas as pd
from pandas import DataFrame
import ast

def separate_last_token(x):
    return (x[:-1], [x[-1]])

#tokenized_df = pd.read_csv('openwebtext6.csv', converters={'input_ids': ast.literal_eval})
    
#vector = tokenized_df['input_ids']
#vector = vector.apply(separate_last_token)

def create_transformer(genome, config):
    """ Creates a TransformerNetwork from a genome and config. """
    return transformer.TransformerNetwork.create(genome, config)

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between true values and predicted values.

    Args:
        y_true (list or numpy array): Actual target values.
        y_pred (list or numpy array): Predicted target values.

    Returns:
        float: Mean squared error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)



def get_data_set():
    #return vector.sample(n = 64)
    return [
        ([0.0, 0.0], [0.0]),
        ([1.0, 0.0], [1.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 1.0], [1.0])
    ]

# Evaluate genome
def eval_genome(genome_list, config):
    fitnesses = []

    for i, genome in genome_list:
        #print("working on genome", i)
        network = create_transformer(genome, config)
        fitness = 0
        for inputs, expected_output in get_data_set():  # Replace `dataset` with your task's data
            output = network.activate(inputs)
            fitness -= mean_squared_error(expected_output, output)  # Example fitness
        fitnesses.append(fitness)
        genome.fitness = fitness

    return fitnesses

# Load the configuration
config = neat.Config(
    neat.DefaultGenome,  # Ensure this is the correct genome class
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-transformer'
)

# Create the population
population = neat.Population(config)

# Add reporters
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

# Run the NEAT algorithm
winner = population.run(eval_genome, n=100)

# Save the best genome
with open('winner-genome.pkl', 'wb') as f:
    pickle.dump(winner, f)

if winner:
    # You can visualize the winner's network using visualize.draw_network()
    draw_net(config, winner, view=True, filename='XOR-network_visualization')

    # Optionally, visualize the population's average fitness over generations
    plot_stats(stats, filename='XOR-fitness_over_generations')

    # Plot species' fitness
    plot_species(stats, filename='XOR-species_over_generations')