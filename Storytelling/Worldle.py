# Worldle.py

# Construct a game world, and let NPCs acting as PCs
# V0.1 simple world and simple rules for NPCs.

# Attrs: Name, Age, Gender, HP, ATK, [relations], etc
# Actions: Move, Talk, Fight(Kill), Show Love, Marry, Find(Objs)
# Locations: Homes(All NPCs'), Village, Wild
# NPCs: 3 random NPCs
# Objs: Treasure, Weapon, Poison, BE, HE.

# Game loop:
#   -   Each Turn NPCs randomly move to a location, and time randomly passed
#   -   While two NPCs are at same location, randomly choose an action
#   -   While only single NPC at a location, trying find sth with some probability
# Game End:
#   -   NPC will die while killed in <Fight>
#   -   NPC will die while reached lifespan
#   -   Babies would be born while <Married> and <Time passed>
#   -   Game would be end while only one NPC left, or find BE, HE objs.

import random

class NPC:  # 
    def __init__(self):
        pass