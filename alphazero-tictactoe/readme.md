[alphazero_algorithm]: alphazero_algorithm.png
[MCTS_summary]: MCTS_summary.png

# About Alphazero

![Alphazero][alphazero_algorithm]


AlphaZero's algorithm is a deep reinforcement learning algorithm that uses neural networks to learn to play games without being pre-programmed with specific strategies. 

The general algorithm for AlphaZero can be summarized as follows:

1. Initialize the game's rules and starting state.
2. Create a neural network to represent the game's state and to make predictions for each legal move.
3. Use Monte Carlo tree search to explore the game tree and to select the most promising moves.
4. Use the predictions from the neural network to refine the selection of moves and to bias the search toward more promising areas of the game tree.
5. Play out the selected moves to the end of the game and determine the outcome.
6. Use the outcome to update the neural network and to improve its performance.
7. Repeat steps 2-6 for many iterations until the neural network can play the game at a high level of performance. 

AlphaZero's algorithm is designed to learn from scratch, taking in only the rules of the game and no human domain knowledge or data. By using deep reinforcement learning, AlphaZero is able to generalize across different games and achieve superhuman performance in games like chess, shogi, and Go.

# About MCTS

![MCTS][MCTS_summary]


Monte Carlo Tree Search (MCTS) is a search algorithm used primarily in game-playing AI systems. The algorithm involves the following steps:

1. `Selection`: Starting from the root node, selection involves choosing a child node that maximizes an exploration-exploitation trade-off criterion (most commonly a function called Upper Confidence Bounds for Trees or UCT). The node is selected repeatedly until a leaf node is reached.
2. `Expansion`: Once a leaf node is reached, a new child node representing an unexplored move is added to the tree. The state of the new node is determined by applying the chosen move to the parent node's state.
3. `Simulation`: A simulation is performed from the newly added node to the end of the game. This simulation involves playing out the game to the end using a random policy to determine the outcome.
4. `Backpropagation`: The outcome of the game is propagated back up the tree from the newly added node to the root node. The total outcome is incremented for all nodes visited during the selection phase.

This process is repeated many times (often tens of thousands or millions of times) to build a tree of board states and associated values, which is then used to choose the best move to make. The more iterations the algorithm goes through, the better its performance is likely to be. The end result is expected to be a more accurate representation of the game tree, which can then be used to create a good AI player.