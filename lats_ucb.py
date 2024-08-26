import os
import random
import logging
from openai import AzureOpenAI
from graphviz import Digraph

# Set the environment variables
os.environ["AZURE_OPENAI_API_KEY"] = "XXX"
os.environ["AZURE_OPENAI_ENDPOINT"] = "XXX"

# Retrieve the environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")

deployment = 'gpt-4o-XXX'

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-06-01",
    azure_endpoint=AZURE_OPENAI_API_ENDPOINT
)

# Setup logging
logging.basicConfig(filename='investment_trace.txt', level=logging.INFO, format='%(message)s')

# Setup logging to redirect to the notebook cell output
class NotebookHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        print(f'**{log_entry}**')

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear previous handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Add the notebook handler to the logger
notebook_handler = NotebookHandler()
notebook_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(notebook_handler)

# Define the Node class
class Node:
    def __init__(self, value, description="", parent=None):
        self.value = value
        self.description = description
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0  # Initial score is 0 until simulated
        self.node_id = str(id(self))  # Unique identifier for graph visualization

    def add_child(self, child):
        self.children.append(child)
        logging.info(f"Added child with description: {child.description}")

    def update_score(self, reward):
        self.score = reward  # Assign the reward as the node's score
        self.visits += 1
        logging.info(f"Updated node {self.description} with reward {reward}")
        if self.parent:
            parent_best_score = max(child.score for child in self.parent.children if child.score is not None)
            self.parent.update_score(parent_best_score / 2)  # Propagate the best reward up with diminishing effect

    def backpropagate(self):
        current_node = self
        while current_node.parent is not None:
            # Update parent's score based on the average of its children's scores
            current_node.parent.score = sum(child.score for child in current_node.parent.children) / len(current_node.parent.children)
            logging.info(f"Backpropagated to parent node {current_node.parent.description}, updated score: {current_node.parent.score}")
            current_node = current_node.parent

# Define the LATS class
class LATS:
    def __init__(self, house_name, client, depth=5):
        self.house_name = house_name
        self.client = client
        self.root = Node("root", f"Investment decisions for {house_name} based on IMF report")
        self.current_node = self.root
        self.depth = depth
        self.graph = Digraph(comment=f'{house_name} Investment Decision Tree')
        self.graph.attr(size='12,12')  # Set graph size for better spacing
        self.graph.node(self.root.node_id, f"{self.root.description}\nScore: N/A", shape='ellipse', style='filled', color='lightblue')
        self.optimal_path = []  # Store the optimal path
        self.thinking_traces = []  # Store all thinking traces

    def load_context(self):
        with open('IMF_context.txt', 'r') as f:
            return f.read()

    def expand_options(self, node):
        # Generate sub-strategies using the LLM
        context = self.load_context()
        strategy_context = (
            f"Given the IMF report context: {context}, "
            f"the current investment strategy is: {node.description}. "
            "What are the next possible investment strategies to consider?"
        )
        messages = [
            {"role": "system", "content": "Generate investment strategies within the context of global economic conditions."},
            {"role": "user", "content": strategy_context}
        ]
        completion = self.client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=150
        )
        sub_strategies = completion.choices[0].message.content.split('\n')
        for strategy in sub_strategies:
            if strategy.strip():
                description = f"{self.house_name} decides to {strategy.strip()}"
                new_node = Node(strategy.strip(), description, node)
                node.add_child(new_node)
                self.graph.node(new_node.node_id, f"{new_node.description}\nScore: N/A", shape='ellipse', style='filled', color='lightgray')
                self.graph.edge(node.node_id, new_node.node_id)

    def simulate_outcomes(self, node):
        context = self.load_context()
        outcome_context = (
            f"Given the IMF report context: {context}, "
            f"the strategy being considered is: {node.description}. "
            "Please simulate the possible outcomes of this strategy."
        )
        messages = [
            {"role": "system", "content": "Simulate this investment strategy within the context of global economic conditions."},
            {"role": "user", "content": outcome_context}
        ]
        completion = self.client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=150
        )
        outcome = completion.choices[0].message.content
        logging.info(f"Simulated outcome for node {node.description}: {outcome}")
        return outcome

    def evaluate_decision(self, outcome):
        if "victory" in outcome.lower() or "success" in outcome.lower():
            return random.randint(70, 100)
        elif "failure" in outcome.lower() or "loss" in outcome.lower():
            return random.randint(1, 30)
        else:
            return random.randint(31, 69)

    def run_simulation_and_backpropagation(self, node):
        outcome = self.simulate_outcomes(node)
        score = self.evaluate_decision(outcome)
        node.update_score(score)
        node.backpropagate()
        return score

    def make_strategic_decision(self):
        self.expand_options(self.current_node)  # Corrected line to pass the node
        best_score = -1
        best_node = None

        for child in self.current_node.children:
            if not child.description or "Thinking Trace" in child.description:
                continue  # Skip nodes without valid strategies

            self.current_node = child
            outcome = self.simulate_outcomes(self.current_node)  # Pass the node here
            thinking_trace = self.generate_thinking_trace(outcome)
            score = self.evaluate_decision(outcome)
            self.current_node.update_score(score)  # Update the score for the current node

            logging.info(f"Strategy: {self.current_node.description}, Outcome: {outcome}, Score: {score}")
            logging.info(f"Thinking Trace: {thinking_trace}")

            self.graph.node(self.current_node.node_id, f"{self.current_node.description}\nScore: {score}", shape='ellipse', style='filled', color='lightgray')

            if score > best_score:
                best_score = score
                best_node = self.current_node

        if best_node:
            self.current_node = best_node  # Move to the node with the best score
            self.highlight_optimal_path(best_node)

    def highlight_optimal_path(self, node):
        while node:
            self.optimal_path.append(node)  # Save the optimal node
            self.graph.node(node.node_id, f"{node.description}\nScore: {node.score}", style='filled', color='green')
            if node.parent:
                self.graph.edge(node.parent.node_id, node.node_id, color='green', penwidth='2')
            node = node.parent

    def run(self):
        for _ in range(self.depth):
            if not self.current_node.children:
                self.expand_options(self.current_node)
            self.make_strategic_decision()

        # After running, display the thinking traces
        with open('investment_trace.txt', 'r') as f:
            for line in f:
                print(line.strip())  # Print the thinking traces to the console

    def display_tree(self):
        self.graph.render('output/investment_decision_tree_with_optimal_path', view=True)
        return self.graph

    def generate_thinking_trace(self, outcome):
        context = self.load_context()
        reflection_context = (
            f"Given the IMF report context: {context}, "
            f"the decision for the investment strategy was taken. "
            f"Reflect on the rationale behind this decision based on the outcome: {outcome}."
        )
        messages = [
            {"role": "system", "content": "Reflect on this investment strategy within the context of global economic conditions."},
            {"role": "user", "content": reflection_context}
        ]
        completion = self.client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=150
        )
        thinking_trace = completion.choices[0].message.content
        self.thinking_traces.append(thinking_trace)  # Collect all thinking traces
        logging.info(f"Thinking Trace: {thinking_trace}")  # Write to log file and display on console
        
        # Ensure the thinking trace does not overwrite the strategy description
        return thinking_trace

    def summarize_optimal_strategy(self):
        optimal_path = []
        full_summary = []

        # Collecting insights from the entire optimal path
        for node in reversed(self.optimal_path):  # Reverse to get correct order
            optimal_path.append(node.description)
            # Include detailed rationale for each decision step
            full_summary.append(f"{node.description}\nScore: {node.score}")

        summary = " -> ".join(optimal_path)

        # Additional LLM call to elaborate on the complete strategy path
        context = (
            f"Given the following investment strategy path: {summary}. "
            "Please provide a concise and insightful summary of each step and its significance."
        )
        messages = [
            {"role": "system", "content": "Summarize this optimal investment strategy within the context of global economic conditions."},
            {"role": "user", "content": context}
        ]
        completion = self.client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=300  # Allow more tokens to capture a comprehensive summary
        )
        detailed_summary = completion.choices[0].message.content

        # Append the full collected summary for complete context
        detailed_summary += "\n\nFull Strategy Path Summary:\n" + "\n".join(full_summary)

        logging.info(f"Optimal Strategy Path: {detailed_summary}")
        with open('optimal_strategy.txt', 'w') as f:
            f.write(detailed_summary)
        return detailed_summary

# Example usage
if __name__ == "__main__":
    house_name = "Investment Advisor"
    advisor = LATS(house_name, client, depth=5)
    
    # Run the strategy simulation
    advisor.run()
    
    # Display the decision tree with thinking traces and optimal path highlighted
    advisor.display_tree()
    
    # Summarize the optimal strategy with detailed steps
    optimal_strategy_summary = advisor.summarize_optimal_strategy()
    print("Optimal Strategy Summary:", optimal_strategy_summary)
