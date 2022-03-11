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

class NPC:  # Try to make Characters live, NPC has attrs like name, age, lifespan, etc
    def __init__(self, name='吴茗', age=18, gender='Male', hp=100,  atk=23, lifespan=73, favor={'吴茗':100}):
        self.name = name
        self.age = age
        self.gender = gender
        self.hp = hp
        self.atk = atk
        self.lifespan = lifespan
        self.favor = favor  # relationship with other NPC

def game(): # One game loop
    is_game_end = False
    while (is_game_end == False):
        pass