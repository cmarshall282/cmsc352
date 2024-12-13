import warnings
import graphviz
import matplotlib.pyplot as plt
import numpy as np

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """Plots the population's average and best fitness."""
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.figure()
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    """Visualizes speciation throughout evolution."""
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True,
             prune_unused=False, node_colors=None, fmt='svg'):
    """Receives a genome and draws a neural network with arbitrary topology."""
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # Prune unused nodes if requested
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    if node_colors is None:
        node_colors = {}

    dot = graphviz.Digraph(format=fmt, node_attr={'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'})

    inputs = set(config.genome_config.input_keys)
    for k in inputs:
        name = node_names.get(k, str(k))
        dot.node(name, _attributes={'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')})

    outputs = set(config.genome_config.output_keys)
    for k in outputs:
        name = node_names.get(k, str(k))
        dot.node(name, _attributes={'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')})

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n not in inputs and n not in outputs:
            dot.node(str(n), _attributes={'style': 'filled', 'fillcolor': node_colors.get(n, 'white')})

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
