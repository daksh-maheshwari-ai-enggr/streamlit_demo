from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from typing import TypedDict, Optional, Dict
from dotenv import load_dotenv
import json
import re

load_dotenv()

class ChefState(TypedDict):
    image_url: str
    budget: int
    nutrition_goal: str
    inventory: Optional[Dict]


llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)

def vision_node(state):
    image_url = state["image_url"]

    message = HumanMessage(
        content=[
            {"type": "text", "text": """
            Extract all visible food items.
            Return ONLY JSON format like:
            {
              "item_name": {"condition": "fresh/wilted", "approx_quantity": "small/medium/large"}
            }
            """},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ]
    )

    response = llm.invoke([message])

    content = response.content

    json_match = re.search(r"\{.*\}", content, re.DOTALL)

    if json_match:
        inventory = json.loads(json_match.group())
    else:
        inventory = {"error": "Invalid JSON response"}

    return {"inventory": inventory} 

# main.py

from langgraph.graph import StateGraph, END

builder = StateGraph(ChefState)

# Add Node
builder.add_node("vision", vision_node)

# Entry point
builder.set_entry_point("vision")

# End after vision for now
builder.add_edge("vision", END)

graph = builder.compile()