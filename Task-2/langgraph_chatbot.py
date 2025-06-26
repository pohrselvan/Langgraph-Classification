from typing import TypedDict
from langgraph.graph import StateGraph , END, START
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Image, display
import logging 


logging.basicConfig(
    filename="pipeline.log",
    filemode="a",  
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class AgentState(TypedDict):
  message : str 
  result : str 
  confidence_score : float 
  HummanorAI : int


model = AutoModelForSequenceClassification.from_pretrained("Task-2/results/checkpoint-3750")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
 

# Inference Node 

def inference_node(state : AgentState) -> AgentState:
  """classification of the text either its positive or negative"""
  tokens = tokenizer(state["message"],padding="max_length",truncation=True,max_length=216,return_tensors="pt")
  with torch.no_grad():
    logits = model(**tokens).logits
    probs = F.softmax(logits,dim=1)
    confidence_score, pred_class = torch.max(probs,dim=1)
  state["confidence_score"] = confidence_score[0]*100
  id2labels = {
    1:"positive",
    0:"negative"
  }
  state["result"] = id2labels[pred_class.item()]

  return state

# Confidence checking Node 


def confidence_score_check(state:AgentState):
  """To Check the confidence score and redirect the path"""
  if state["confidence_score"] < 60.000:
    logging.WARN("FallBack mechanism as trigged")
    return "FallBack_Mechanism"
  else:
    return "End"
  
# fallback router 


def fallback_router(state:AgentState):
  """To check use wether Ai or human"""
  if state["HummanorAI"] == 0:
    logging.WARN("Human fallback mechanism as trigged")
    return "Human_fallback"
  else: 
    logging.WARN("AI FallBack mechanism as triggered")
    return "AI_fallback"
  
# human fallback  Node 


def human_fallback(state:AgentState):
  print(f"\n [Human_FallBack]Prediction was Low : '{state['result']}' with score {state['confidence_score']:.2f}")
  logging.WARN(f"[Human_FallBack]Prediction was Low : '{state['result']}' with score {state['confidence_score']:.2f}")

  user_input = input("[Human_FallBack]Was the prediction is correct(y/n)?:")
  logging.INFO(f"[Human_FallBack]Was the prediction is correct(y/n)?:{user_input}")
  if user_input.lower() == "y":
    print("\n[Human_FallBack]Thank you for the verification")
    logging.INFO(f"[Human_FallBack]Thank you for the verification")
  else: 
    corrected = input("[Human_FallBack]What is the correct label:")
    state["result"] = corrected 
    logging.INFO(f"[Human_FallBack]What is the correct label:{corrected}")
  return state 

# AI fallback 


def ai_fallback(state:AgentState):
  logging.INFO("Loading the Model.............")
  llm = Ollama(model = "llama3.2") 
  prompt = ChatPromptTemplate(
      [
          ("system", "You are an expert text classification assistant. Your task is to classify each input strictly as either 'Positive' or 'Negative'. Do not provide explanations, just output the predicted label as either 'Positive' or 'Negative' based on the input text sentiment."),
          ("user","Questions : {question}")
      ]
  )

  output = StrOutputParser()

  # declaring the chain
  chain = prompt|llm|output

  result = chain.invoke({"question":state["message"]})

  print(f"\n[AI_FallBack]Corrected output by llama model is : {result}")
  logging.WARN(f"\n[AI_FallBack]Corrected output by llama model is : {result}")

  state["result"] = result

  return state

# start the graph structure 
graph = StateGraph(AgentState)

graph.add_node("Inference_Node",inference_node)
graph.add_node("Confidence_Score_Check",lambda state : state)
graph.add_node("fallback_mechanism",lambda state : state)

# lets build connection 
graph.add_edge(START,"Inference_Node")
graph.add_edge("Inference_Node","Confidence_Score_Check")

graph.add_conditional_edges(
  "Confidence_Score_Check",
  confidence_score_check,
  {
    "FallBack_Mechanism" : "fallback_mechanism",
    "End":END
  }
)

# adding conditional to fallback mecahanism 
graph.add_node("Human_Fallback",human_fallback)
graph.add_node("AI_Fallback",ai_fallback)
graph.add_conditional_edges(
  "fallback_mechanism",
  fallback_router,
  {
    "Human_fallback" : "Human_Fallback",
    "AI_fallback": "AI_Fallback"
  } 
)

graph.add_edge("Human_Fallback",END)
graph.add_edge("AI_Fallback",END)

app = graph.compile()

#visualize the graph 

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     print("can't able to print")
print("Welcome to the text classification chatbot\n")
print("Note: Please Enter the fallback Mechanism you want")
logging.info("Started the classification pipeline")

human_or_ai = int(input("0 -> Human or 1-> AI:"))
logging.info(f"0 -> Human or 1-> AI: {human_or_ai}")

print("starting test classification pipeline with LANGGRAPH....................................................")
user_inputs = input("User: ")
while user_inputs.lower() != "exit":
    logging.info(f"User Inputs: {user_inputs}")
    inputs = AgentState(message=user_inputs, HummanorAI=human_or_ai)
    result = app.invoke(inputs)
    final_label = result["result"]
    logging.info(f"Final lable: {final_label}")
    print("Final Label:",result["result"])
    user_inputs = input("User: ")

print("Ending....................................................................")
logging.info("Stopping the text classification pipeline")
