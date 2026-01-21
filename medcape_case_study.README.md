# Medscape Take-Home Assignment: Agentic AI Workflow Challenge Alejandro Rodas

## Introduction 
In this project, you'll find two solutions to the challenge. I wanted to present two different scripts so I can answer the question to how is the solution scalable and more robust 
to handle production level operations. 

## Files

The first py script called solution_AIAgent and the second py script production_solution_AIAgent.

## Design Choices: 

### What you'll find in AI_workflow_solution:

2 Agents (analytics_agent and retriever_agent)

-Analytics Agent with the following tools for quick calculation of relevant data: "get_roi", "get_cpc", "get_cpa", "get_ctr", "get_cvr"

-Retriever Agent with the following tool for fetching relevant data: "retriever_tool"

Helper funcitons like:

"should_continue" and "router" that respectively check if the last message contains tool calls. And the second, where  based on the selected tool redirects to the appropriate agent. 

Where the graph representing the orchestration looks like this: 

<img width="481" height="273" alt="image" src="https://github.com/user-attachments/assets/6bacf0ac-9560-409e-9f23-c8e199798ac2" />

### What you'll find in production_solution_AIAgent:

The same Agents and Tools and before (Analytics Agent with "get_roi", "get_cpc", "get_cpa", "get_ctr", "get_cvr" and a Retriever Agent with a "retriever_tool")

This time with an ML Agent, supporting three ML models: K-Means, Logistic Regression and Random Forests deeming them the most appropriate for this project ("random_forest", "k_means", "log_reg")
and a Visualization Agent for easier understanding of the models decisions and calculations ("aggregate_xy", "plot_line", "plot_bar"). Enabled through matplotlib.

Alongside with the other two helper functions: 

-A 'reflection_node' was added for a more elaborate response and to limit error/hallucination as much as possible (LLM was also set at 'temperature' 0 to limit hallucination as well).

Finally, two key design choices were made to make this friendler with bigger knowledge bases/production level code: 

-The knowledge base was uploaded to BigQuery and retrieved through an API call, so it can be queried with the freshest data,  making it more adaptable with knowledge bases that not only could be bigger
but constantly modified/updated. 

-ChromaDB was also used as it makes the retriever scalable by storing embeddings persistently, avoiding full KB reloads/scans as the knowledge base grows.
It also enables fast semantic + metadata filtered retrieval behind a stable interface, so data sources/backends can be swapped without rewriting the workflow.

Where the graph representing the orchestration of production_solution_AIAgent looks like this: 
<img width="896" height="357" alt="image" src="https://github.com/user-attachments/assets/65116aad-9407-4002-8543-a28250b1551a" />


### The system's architecture is as follows: 
To summarize (as you can see the full context in the Design Choices section) the final orchestration would be the following: 

AI_workflow_solution:

A Controller (LLM) receives a natural-language query and routes it via router/should_continue to either the Retriever Agent (retriever_tool) for relevant KB context, the Analytics Agent (ROI/CPC/CPA... etc tools) for deterministic computations, or both. Then, the Controller  synthesizes the retrieved context with computed metrics into a structured recommendation.

production_solution_AIAgent: 

The same core routing + Retriever/Analytics flow, extended with optional ML + Visualization agents for deeper analysis and interpretability, plus a reflection_node that reviews tool outputs and the draft response for consistency before returning the final recommendation. Retrieval this time, is made through a production-friendly via BigQuery-backed data access and ChromaDB vector retrieval for persistent, scalable semantic search.

## How to make this even more scalable/limitations: 
Some agents could be concurrent, for instance: parallelize retrieval + analytics (and ML/visualization) when both are needed, this cut latency as workload grows. Also, more agents could be used/created + having more robust tools that could work with unstructured data. Though these tools and this specific level of robustness was chosen to demostrate how Agents would work with this knowledge base and to not compromise latency.

