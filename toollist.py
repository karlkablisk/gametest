import random
from typing import Tuple, Optional
from langchain.tools import tool

ALL_TOOLS = []

class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description
        ALL_TOOLS.append(self)

class SerpAPIWrapper:
    def run(self):
        return "some data"

# Create Search tool and automatically add it to ALL_TOOLS
search = SerpAPIWrapper()
Tool(
    name="Search",
    func=search.run,
    description="Useful for when you don't know the answer and need to find more information online.",
)

    
# Tool function
def discover_item_tool(user_context: str, difficulty: int) -> str:
    item_name = f"{user_context}_item_{random.randint(1, 100)}"
    item_description = f"A special item found in {user_context}"

    rarities = ['Common', 'Uncommon', 'Rare', 'Epic', 'Legendary']
    rarity = random.choices(rarities, weights=[50, 30, 15, 4, 1])[0]

    altered_stats_count = random.randint(1, 5)
    player_stats = ['Health', 'Speed', 'Strength', 'Agility', 'Magic']
    altered_stats = random.sample(player_stats, altered_stats_count)
    
    multiplier = 10 * (difficulty - 1)
    stat_values = [random.randint(-10, 10) * multiplier for _ in range(altered_stats_count)]

    return item_name, item_description, rarity, altered_stats, stat_values
    
    # Class for storing item information
class ItemInfo:
    def __init__(self, item_name, item_description, rarity, altered_stats, stat_values):
        self.item_name = item_name
        self.item_description = item_description
        self.rarity = rarity
        self.altered_stats = altered_stats
        self.stat_values = stat_values
    
    def get_info(self):
        return self.item_name, self.item_description, self.rarity, self.altered_stats, self.stat_values

# Tools list
tools = [
   Tool(
       name=f"discover_item",
       func=discover_item_tool,
       description=f"A function to discover items based on the current situation in the user's context.",
   )
]


# Construct the string representation
#tool_names = [f"[{tool.name}]" for tool in ALL_TOOLS]
#tools_string = ' + '.join(tool_names)
tool_names = ', '.join([tool.name for tool in ALL_TOOLS])

def tools_string():
    return ", ".join([tool.__name__ for tool in ALL_TOOLS])


def get_all_tools():
    return ALL_TOOLS
#print(tools_string)

