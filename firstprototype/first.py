from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from typing import TypedDict, Optional, Dict,List
from dotenv import load_dotenv
import json
import re

load_dotenv()

class ChefState(TypedDict):
    image_url: str
    budget: int
    nutrition_goal: str
    inventory: Optional[Dict]
    risk_items: Optional[List]


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

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import json



def detect_risk(inventory):

    prompt = f"""
You are a food waste expert.

Given the fridge inventory below, identify foods that are likely to spoil soon.

Return JSON format:

{{
 "at_risk":[
   {{"item":"name","reason":"why it spoils fast"}}
 ]
}}

Inventory:
{inventory}
"""

    message = HumanMessage(content=prompt)
    response = llm.invoke([message])

    content = response.content

    json_match = re.search(r"\{.*\}", content, re.DOTALL)

    if json_match:
        data = json.loads(json_match.group())
        risk_items = data.get("at_risk", [])
    else:
        risk_items = []

    return {
        "risk_items": risk_items
    }


# main.py

from langgraph.graph import StateGraph, END

builder = StateGraph(ChefState)

# Add Node
builder.add_node("vision", vision_node)
builder.add_node("risk",detect_risk)

# Entry point
builder.set_entry_point("vision")

# End after vision for now
builder.add_edge("vision","risk")
builder.add_edge("risk",END)


graph = builder.compile()