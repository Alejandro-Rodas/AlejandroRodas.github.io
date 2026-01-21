import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, TypedDict, List
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.document_loaders import CSVLoader
from typing import Any
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_core.messages import AIMessage
import json
from google.cloud import bigquery
llm = ChatOpenAI(
    model = "gpt-4o", temperature=0
)



embeddings = OpenAIEmbeddings(
    model= "text-embedding-3-small"
)

#if not os.path.exists(csv_path):
#    raise FileNotFoundError(f"CSV file not found: {csv_path}")


# Checks if the CSV is there
#try:
#    pages = csv_loader.load()
#    print(f"CSV has been loaded and has {len(pages)} rows")
#except Exception as e:
#    print(f"Error loading CSV: {e}")
#    raise


from functools import lru_cache
from google.cloud import bigquery

BQ_TABLE = "medscape-case-study.1234_.campaign_performance"
bq_client = bigquery.Client(project="medscape-case-study")

@lru_cache(maxsize=1)
def load_campaign_df():
    return bq_client.query(f"SELECT * FROM `{BQ_TABLE}`").to_dataframe()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

from langchain_community.document_loaders import DataFrameLoader

df = load_campaign_df().copy()
df["_text"] = df.astype(str).agg(" | ".join, axis=1)

pages = DataFrameLoader(df, page_content_column="_text").load()


os.environ["CHROMA_CLOUD_API_KEY"] = "ck-BnpGsmd3uMjaerHcx3svNT2pBo5GTuD9sizQg8xMSKqH"
os.environ["CHROMA_TENANT"] = "b929d836-ce14-4425-ba00-69db120fbd3d"
os.environ["CHROMA_DATABASE"] = "MedScapeDB"


import chromadb

client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_CLOUD_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
)
collection = client.get_or_create_collection("campaign_performance")


import os
import chromadb
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_CLOUD_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
)

vectorstore = Chroma(
    client=client,
    collection_name="campaign_performance",
    embedding_function=embeddings,
)

# one-time ingestion
vectorstore.add_documents(pages_split)

retriever = vectorstore.as_retriever(search_kwargs={"k": 7})


#Retriever Tools
@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Campaign Performance document.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in  the Campaign Performance document."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}") 
    return "\n\n".join(results)


#Anlytics Tools
@tool
def get_roi(query: Optional[dict] = None) -> str:
    """Return on Investment (ROI) = (revenue - spend) / spend. Expects query={'revenue': ..., 'spend': ...}."""
    if not query:
        return "Missing input for ROI. Expected {'revenue': ..., 'spend': ...}."
    revenue = query.get("revenue")
    spend = query.get("spend")
    if revenue is None or spend is None:
        return "Missing revenue/spend for ROI."
    if spend == 0:
        return "Spend is 0; ROI undefined."
    return str((revenue - spend) / spend)
#@tool
#def get_cpc(query: Optional[dict] = None) -> str:
#    """Cost per click. Expects query={'spend': ..., 'clicks': ...}."""
#    if not query or "spend" not in query or "clicks" not in query:
#        return "Missing spend/clicks for CPC."
#    if query["clicks"] == 0:
#        return "Clicks is 0; CPC undefined."
#    return str(query["spend"] / query["clicks"])
@tool 
def get_cpc(query: Optional[dict] = None) -> str: 
    """Cost per Click (CPC) = spend/ clicks. Expects query={'spend': ..., 'clicks': ...}."""
    if not query:
        return "Missing input for CPC. Expected {'clicks': ..., 'spend:...'}"
    clicks = query.get("clicks")
    spend = query.get("spend")
    if spend is None or clicks is None: 
        return "Missing revenue or clicks for CPC"
    if spend == 0:
        return "NaN value for spend"
    if clicks == 0:
        return "NaN value for clicks"
    return str (spend/clicks)

@tool 
def get_cpa(query: Optional[dict] = None) -> str: 
    """Cost per Aquisition. Exptects query={'spend': ...., 'conversions':...}"""
    if not query or 'spend' not in query or 'conversions' not in query: 
        return "Missing inputs for correct calculation"
    spend = query.get('spend')
    conversions = query.get('conversions')
    if spend is None or conversions is None: 
        return "Missing either spend or conversions"
    if spend == 0: 
        return "Spend has a null value (0)"
    if conversions == 0:
        return "conversions has a null value (0)"
    return str(spend/conversions)

#@tool 
#def get_cpa(query: Optional[dict] = None) -> str:
#    """Cost per Aquisition. Expects query={'spend': ..., 'conversions': ...}."""
#    if not query or "spend" not in query or "conversions" not in query:
#        return "Missing spend/conversions for CPA."
#    if query["clicks"] == 0:
#        return "Clicks is 0; CPC undefined."
#    return str(query ["spend"]/ query["conversions"])


@tool
def get_ctr(query: Optional[dict] = None) -> str: 
    """Click through rate . Expects query={'impressions': ..., 'clicks': ...}."""
    if not query or 'clicks' not in query or 'conversions' not in query: 
        return "Missing inputs for correct calculation"
    impressions = query.get('impressions')
    clicks = query.get('clicks')
    if clicks is None or impressions is None: 
        return "Missing either spend or conversions"
    if impressions == 0: 
        return "The campaign had 0 impressions"
    if clicks == 0:
        return "The campaign had 0 clicks"
    return str((clicks/impressions) * 100)

@tool
def get_cvr(query: Optional[dict] = None) -> str: 
    """Conversion Rate. Expects query={'conversions': ..., 'clicks': ...}."""
    if not query or 'clicks' not in query or 'conversions' not in query: 
        return "Missing inputs for correct calculation"
    clicks= query.get('clicks')
    conversions = query.get('conversions')
    if clicks is None or conversions is None: 
        return "Missing either clicks or conversions"
    if clicks == 0: 
        return "clicks has a null value (0)"
    if conversions == 0:
        return "conversions has a null value (0)"
    return str((conversions/clicks) * 100)

#Visualization Tools

@tool 
def aggregate_xy(query: Optional[Dict[str, Any]] = None) -> str:
    """
    Combine rows that share the same x (and optional group) into one value of y.
    agg could have the following operations"sum" | "mean" | "count"
    and returns: list of dict rows (same format as input), aggregated.
    """
    if not data:
        return []
    df = pd.DataFrame(data)

    if x not in df.columns or (agg != "count" and y not in df.columns):
        return []
    keys = [x]
    if group: 
        keys.append(group)
        
    if agg == "mean": 
        out = df.groupby(keys, dropna=False)[y].mean().reset_index()
    elif agg == "count":
        out = df.groupby(keys, dropna=False).size().reset_index(name=y)
    else: 
        out = df.groupby(keys, dropna=False)[y].sum().reset_index()

    return out.to_dict(orient="records")

@tool 
def plot_bar(data: Optional[List[Dict[str, Any]]], x: str, y: str, title: str = "", top_n: int = 0) -> str:
    """
    Plot a simple bar chart of y by x.
    - If top_n > 0, keep only the top_n rows by y.
    """
    
    if not data:
        return f"Missing data"
    df = pd.DataFrame(data)

    if x not in df.columns or y not in df.columns:
        return f"Missing columns"

    df = df.sort_values(y, ascending=False)
    if top_n > 0: 
        df = df.head(int(top_n))


    plt.figure()
    plt.bar(df[x].astype(str), df[y])
    plt.title(title or f"{y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()
    return "Rendered bar chart."

@tool 
def plot_line(data: List[Dict[str, Any]], x: str, y: str, title: str = "") -> str:
    """
    Plot a simple line chart of y over x.
    Best when x is time-like (date).
    """
   
    if not data:
        return f"Missing data"
    df = pd.DataFrame(data)

    if x not in df.columns or y not in df.columns:
        return f"Missing columns"
    
    df = df.sort_values(y, ascending=False)
    try:
      df[x] = pd.to_datetime(df[x])
    except Exception:
      pass

    plt.figure()
    plt.plot(df[x], df[y])
    plt.title(title or f"{y} over {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()
    return "Rendered line chart."


#DS/ML Tools
@tool 
def random_forest(_: dict | None = None) -> dict:
    """
    Train a lightweight RandomForest model to predict campaign ROI from the mock dataset.

    ROI is defined as revenue / spend. The function:
      - loads /mnt/data/campaign_performance.csv
      - builds a simple feature set (specialty, tactic, quarter, impressions, clicks, conversions, spend)
      - one-hot encodes categorical columns
      - fits a RandomForestRegressor
      - returns a basic holdout RMSE for quick validation

    Returns:
        dict: {"rmse": float} where rmse is the root-mean-squared error on a 20% test split.
    """
    df = load_campaign_df().copy()
    df["roi"] = df["revenue"] / df["spend"]
    X = pd.get_dummies(df[["specialty","tactic","quarter","impressions","clicks","conversions","spend"]], drop_first=True)
    y = df["roi"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=8)
    m = RandomForestRegressor(n_estimators=100, random_state=7).fit(Xtr, ytr)
    rmse = mean_squared_error(yte, m.predict(Xte), squared=False)
    return {"rmse": float(rmse)}

@tool 
def k_means(_: dict | None = None) -> dict:
    """
    Run K-means clustering to segment campaigns by basic performance behavior.

    Loads the campaign dataset, derives a few ratio features (CTR, CVR, ROI) that are also found in the analytics agent fitting a  KMeans model,
    and returns counts per cluster.
    some of the arguments include:
    query: Optional dict (unused), kept for tool-call consistency.
    and returns a dictionary in such manner:
    dict: {"cluster_counts": {cluster_id: count, etc}}
    """
    df = load_campaign_df().copy()
    df["ctr"] = df["clicks"]/df["impressions"]
    df["cvr"] = df["conversions"]/df["clicks"]
    df["roi"] = df["revenue"]/df["spend"]
    X = df[["ctr","cvr","roi","spend"]].fillna(0)
    labels = KMeans(n_clusters=4, random_state=7, n_init="auto").fit_predict(X)
    return {"cluster_counts": df.assign(cluster=labels)["cluster"].value_counts().to_dict()}


@tool
def log_reg(_: dict | None = None) -> dict:
    """
    This tool trains a simple logistic regression classifier for "high ROI" campaigns.
    Defines high ROI as ROI > 1.5, one-hot encodes X  categorical features, fits a
    LogisticRegression model, and returns holdout accuracy.
    as arguments it takes a:
    query: Optional dict (unused), kept for tool-call consistency.
    And then returns a dictionary: 
    dict: {"acc": float} accuracy on a 20% test split.
    """
    df = load_campaign_df().copy()
    df["roi"] = df["revenue"]/df["spend"]
    y = (df["roi"] > 1.5).astype(int)
    X = pd.get_dummies(df[["specialty","tactic","quarter","impressions","clicks","conversions","spend"]], drop_first=True)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=7)
    m = LogisticRegression(max_iter=2000).fit(Xt, yt)
    return {"acc": float(accuracy_score(yt, m.predict(Xt)))}



# After defining ALL tools (retriever_tool + analytics tools + ds tools + visualization tools)
tools = [retriever_tool, get_roi, get_cpc, get_cpa, get_ctr, get_cvr, aggregate_xy, plot_line, plot_bar, random_forest, k_means,log_reg] 
tools_dict = {t.name: t for t in tools}
llm = llm.bind_tools(tools)

#Instantiate Class
class AgentState(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages]


system_prompt = """
You are an  AI assistant who answers questions about Campaign Performance of MedScape in recent advertisement campaigns based on a csv file loaded into your knowledge base.
Based on the user's prompt you can decide to use either the retriever tool to answer questions about the campaign data, or analytics to calculate quick operations, additionally you 
can also plot graphs if you deem fit.  You can make multiple calls if needed. 
If you need to look up some information before asking a follow up question, you are allowed to do so. When calling plot_bar or plot_line you MUST pass:
- data: a list of dicts
- x: column name in each dict
- y: column name in each dict

Example:
plot_bar(
  data=[{"Quarter":"2025Q1","ROI":0.25},{"Quarter":"2025Q2","ROI":0.70}],
  x="Quarter",
  y="ROI",
  title="ROI 2025Q1 vs 2025Q2"
)
Your final response should be like this: 
Summary of findings (2–3 sentences).
▪ Suggested next best action (e.g., “Shift spend from Tactic A to Tactic B”)
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} 

#Agents

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}



# Retriever Agent
def retriever_agent_execution(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t["name"]].invoke(t["args"])
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Information succesfully retrieved!")
    return {'messages': results}

#Analytics Agent

def analytics_agent_execution(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'])
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Analytics execution complete!")
    return {'messages': results}

# Data Visualization Agent
def visualization_agent_execution(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'])
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Visualizations rendered!")
    return {'messages': results}

# ML/DS Agent
def ml_agent_execution(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'])
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Visualizations rendered!")
    return {'messages': results}


#Helper Functions + Reflection Node


def router(state: AgentState) -> str: 
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls: 
        return "end"
     
    analytics_tools = {"get_roi", "get_cpc", "get_cpa", "get_ctr", "get_cvr"}
    retriever_tools = {"retriever_tool"}
    visualization_tools = {"aggregate_xy", "plot_line", "plot_bar"}
    ml_tools = {"random_forest", "k_means", "log_reg"}

    for i in tool_calls: 
        if i["name"] == "retriever_tool": 
            return "retriever_agent"
        if i["name"] in ["aggregate_xy", "plot_line", "plot_bar"]: 
            return "visualization_agent"
        if i["name"] in ["random_forest", "k_means", "log_reg"]: 
            return "ml_agent"
        else: 
            return "analytics_agent"
    return "end"


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


def reflection_node(state: AgentState) -> dict:
    draft = state["messages"][-1].content 
    evaluate_prompt = {
      "draft": draft,
      "rubric": [
        "Do the metrics match the formulas? (ROI = (revenue-spend)/spend, etc.)",
        "Any divide-by-zero or missing-field issues?",
        "Does the recommendation follow from the numbers?",
        "Edge cases: check divide-by-zero, None/missing fields, negative values, and inconsistent units.",
        "Is the output format exactly: summary (2–3 sentences) + next_best_action?"
      ],
      "instruction": (
        "Return ONLY JSON: "
        "{\"ok\": true} or "
        "{\"ok\": false, \"fixed\": {<same schema as draft>}, \"issues\": [..]}"
      )
    }
    
    raw = llm.invoke(json.dumps(evaluate_prompt)).content
    
    try: 
        answer = json.loads(raw)
        final_text = draft if answer.get("ok") else answer["fixed"]
    except json.JSONDecodeError:
        final_text = draft

    return {"messages": [AIMessage(content=final_text)]}



#Graph 
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", retriever_agent_execution)
graph.add_node("analytics_agent", analytics_agent_execution)
graph.add_node("visualization_agent", visualization_agent_execution)
graph.add_node("ml_agent", ml_agent_execution)
graph.add_node("reflection_node", reflection_node)
graph.add_conditional_edges(
    "llm", 
    router, 
    {
    "retriever_agent": "retriever_agent", 
    "analytics_agent" : "analytics_agent",
    "visualization_agent" : "visualization_agent",
    "ml_agent" : "ml_agent",
    "end": "reflection_node",
    }
)
graph.add_edge("retriever_agent", "llm")
graph.add_edge("analytics_agent", "llm")
graph.add_edge("visualization_agent", "llm")
graph.add_edge("ml_agent", "llm")
graph.add_edge("reflection_node", END)
graph.set_entry_point("llm")

RAG_agent = graph.compile()
