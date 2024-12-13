from neat.graphs import required_for_output
import numpy as np

class TransformerNetwork(object):
    def __init__(self, inputs, outputs, attention_heads, feedforward_units, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.attention_heads = attention_heads
        self.feedforward_units = feedforward_units
        self.node_evals = node_evals

        self.values = {k: 0.0 for k in [*inputs, *outputs]}
        for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
            self.values[node] = 0.0
            for i, w in links:
                self.values[i] = 0.0

    def reset(self):
        self.values = {k: 0.0 for k in self.values}

    def attention(self, query, key, value):
        """Scaled Dot-Product Attention."""
        key_dim = query.shape[-1]
        scores = np.dot(query, key.T) / np.sqrt(key_dim)
        
        # Subtract the maximum score for numerical stability
        scores -= np.max(scores, axis=-1, keepdims=True)
    
        # Compute softmax
        exp_scores = np.exp(scores)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        return np.dot(weights, value)

    def activate(self, inputs):
        if len(inputs) != len(self.input_nodes):
            raise RuntimeError(
                f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}"
            )

        # Initialize node values, including all referenced nodes
        node_values = {
            node: 0.0
            for node in set(
                self.input_nodes + self.output_nodes +
                [n for n, _, _, _, _, links in self.node_evals for n, _ in links]
            )
        }

        # Assign input values
        for i, value in zip(self.input_nodes, inputs):
            node_values[i] = value

        # Compute node activations
        for node, activation, aggregation, bias, response, links in self.node_evals:
            # Compute inputs to the current node
            node_inputs = [node_values.get(i, 0.0) * w for i, w in links]

            # Aggregate inputs and apply bias
            aggregated = aggregation(node_inputs) + bias

            # Apply activation function
            node_value = activation(aggregated * response)

            # If attention mechanism is involved, process it
            if self.attention_heads > 1:
                # Ensure the input size matches required shape
                node_inputs = np.array(node_inputs)
                if node_inputs.size < self.attention_heads:
                    pad_size = self.attention_heads - node_inputs.size
                    node_inputs = np.pad(node_inputs, (0, pad_size), mode='constant', constant_values=0)
                elif node_inputs.size > self.attention_heads:
                    node_inputs = node_inputs[:self.attention_heads]

                queries = node_inputs.reshape(self.attention_heads, -1)
                keys = queries.copy()
                values = queries.copy()

                # Compute attention output
                attention_output = self.attention(queries, keys, values)
                node_value += attention_output.sum()  # Integrate attention output

            node_values[node] = node_value

        # Collect output node values
        return [node_values[o] for o in self.output_nodes]

    @staticmethod
    def create(genome, config):
        genome_config = config.genome_config
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys, genome.connections
        )

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        # Ensure every required node has at least one input
        for node_key in genome.nodes:
            if node_key not in node_inputs:
                node_inputs[node_key] = []  # Initialize with empty links if none exist

        # Build evaluations
        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append(
                (node_key, activation_function, aggregation_function, node.bias, node.response, inputs)
            )

        return TransformerNetwork(
            genome_config.input_keys,
            genome_config.output_keys,
            2,
            64,
            node_evals,
        )

