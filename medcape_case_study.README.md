# Medscape Take-Home Assignment: Agentic AI Workflow Challenge

## What it is
In this project. You'll find two solutions to the challenge. I wanted to present two different solutions so I can answer the question to how is the solution scalable and more robust 
to handle prodcution level operations. 

## Files

 The first py script called solution_AIAgent and the second py script production_solution_AIAgent.

### What you'll find in AI_workflow_solution:

2 Agents (analytics_agent and retriever_agent)
Analytics Agent with the following tools for quick calculation of relevant data: "get_roi", "get_cpc", "get_cpa", "get_ctr", "get_cvr"

Retriever Agent with the following tool for feteching relevant data: "retriever_tool"

Helper funcitons like:
"should_continue" and "router" that respectively Check if the last message contains tool calls and router that based on the selected tool redirects to the appropriate agent. 
Where the graph representing the orchestration looks like this: 

<img width="481" height="273" alt="image" src="https://github.com/user-attachments/assets/6bacf0ac-9560-409e-9f23-c8e199798ac2" />

### What you'll find in production_solution_AIAgent:

The same Agents and tools and before (Analytics Agent with get_roi", "get_cpc", "get_cpa", "get_ctr", "get_cvr" and a Retriever Agent with a "retriever_tool")
but this time with an ML Agent for supporting three ML models (K-Means, Logistic Regression and Random Forests deeming them the most appropriate for this project)
and a Visualization Agent for easier understanding of the models decisions and calculations, using primarily matplotlib.

-A 'reflection_node' was added for a more elaborate response and to limit error/hallucination as more as possible. 

Finally, two key design choices were made to make this more like production level code: 

-The knowledge base was uploaded to BigQuery and retrieved through an API call, so it can be queried with the freshest data and works well with knowledge bases 
that are constantly modified/updated. 

-ChromaDB was also used ChromaDB as it makes the retriever scalable by storing embeddings persistently, avoiding full KB reloads/scans as the knowledge base grows.
It also enables fast semantic + metadata filtered retrieval behind a stable interface, so data sources/backends can be swapped without rewriting the workflow.


Where the graph representing the orchestration of production_solution_AIAgent looks like this: 
<img width="896" height="357" alt="image" src="https://github.com/user-attachments/assets/65116aad-9407-4002-8543-a28250b1551a" />


The general orchestration of script is as follows: 
1. Necessary importation of around 20 libraries.
2. LLM call.
3. A retriever tool from the Retriever Agent that fetches relevant information from the knoeledge base.
4. 6 Analytics



## Usage
Example command.


