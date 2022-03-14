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

from random import *
from select import select

class NPC:  # Try to make Characters live, NPC has attrs like name, age, lifespan, etc
    def __init__(self, name='吴茗', age=18, gender='Male', hp=100,  atk=23, lifespan=73, favor={'吴茗':100}):
        self.name = name
        self.age = age
        self.gender = gender
        self.hp = hp
        self.atk = atk
        self.lifespan = lifespan
        self.favor = favor  # relationship with other NPC
        self.location = self.name + '的家'  # NPC's current location.

npc_a = NPC()
npc_b = NPC(name = '陆小凤', age = randint(28, 42), lifespan= randint(75, 88))
npc_c = NPC(name = '郭襄', age = randint(16, 27), gender = 'Female', lifespan = randint(60, 100))
init_npcs = [npc_a, npc_b, npc_c]
init_locations = ['村镇', '市集', '郊外']
for inpc in init_npcs:
    init_locations.append(inpc.location)

def search_npc_by_name(npcs, name):   # as title.
    for npc in npcs:
        if npc.name == name:
            return npc

def take_two(npc_one, npc_two): # Two npcs do something together.
    print (npc_one.name + ' 与 ' + npc_two.name + ' 进行了一番友好的交谈')    # _TO_BE_UPDATED_

class Game: # Global vars in a game, like year, day, hour, is_end, etc
    def __init__(self, year=0, day=0, hour=0, is_over=False, npcs=[], locs=[]):
        self.year = year
        self.day = day
        self.hour = hour
        self.is_over = is_over
        self.npcs = []
        if len(npcs) > 0:
            for npc in npcs:
                self.npcs.append(npc)
        self.locs = []
        if len(locs) > 0:
            for loc in locs:
                self.locs.append(loc)

    def time_pass_by(self, year=0, day=0, hour=1): # Game time passed
        old_year = self.year
        self.hour += hour
        while (self.hour >= 24):
            self.day += 1
            self.hour -= 24
        self.day += day
        while (self.day >= 365):
            self.year += 1
            self.day -= 365
        self.year += year
        if (len(self.npcs) > 0):    # Check NPCs' lifespan
            for npc in self.npcs:
                npc.age += (self.year - old_year)
                if (npc.age > npc.lifespan):
                    self.npcs.remove(npc)
                    print(npc.name + ' 寿终正寝')

def life_game(): # One game loop
    new_game = Game(npcs=init_npcs)
    while (new_game.is_over == False):  # Not game over
        if randint(1, 100) <= 10: # 10% chance fast pass
            rand_year = randint(1,5)
            rand_day = randint(1, 300)
            rand_hour = randint(1,24)
            new_game.time_pass_by(rand_year, rand_day, rand_hour)
        else:
            new_game.time_pass_by()  # Each turn should cost sometime.
        loc_with_npc = {}
        for loc in init_locations:
            loc_with_npc[loc] = []
        for npc in new_game.npcs:   # NPCs act in turn.        
            npc.location = choice(init_locations)   # Everyone has their destination.
            loc_with_npc[npc.location].append(npc.name)
        for loc in loc_with_npc:
            if len(loc_with_npc[loc]) == 1: # Only one NPC, do something single.
                pass
            elif len(loc_with_npc[loc]) > 1:    # Multiple npcs, they will interact with others.
                names = loc_with_npc[loc]
                for name in names:
                    current_npc = search_npc_by_name(new_game.npcs, name)
                    temp_names = names.copy()
                    temp_names.remove(name) # maybe there've some better way to do this?
                    for tn in temp_names:
                        next_npc = search_npc_by_name(new_game.npcs, tn)
                        if (current_npc != None) & (next_npc != None):
                            take_two(current_npc, next_npc) # Do something with the other npc
        if (len(new_game.npcs) == 1):
            new_game.is_over = True
    print('Game Over!')

life_game()