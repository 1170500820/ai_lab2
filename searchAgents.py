# coding=utf-8
# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        # 剩余的节点
        print self.corners
        self.left_corners = list(self.corners)
        self.left_corners.sort()
        self.l = max(abs(top - 1), abs(right - 1))
        self.s = min(abs(top - 1), abs(right - 1))

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"


        # TODO 可能不需要了
        def relative_coordinate(corner):
            current = self.startingPosition
            return current[0] - corner[0], current[1] - corner[1]

        # 初始状态为与每个剩余节点的相对坐标
        initial_left_corners = list(self.corners)
        initial_left_corners.sort()

        return (self.startingPosition, tuple(initial_left_corners))


    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        return len(state[1]) == 0

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]



            "*** YOUR CODE HERE ***"
            # 判断是否装墙
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if self.walls[nextx][nexty]:
                continue

            # 判断是否达到某一个角落
            new_state = list(state[1])
            for corner in list(state[1]):
                if (nextx, nexty) == corner:
                    new_state.remove((nextx, nexty))

            successors.append((((nextx, nexty), tuple(new_state)), action, 1))

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"

    # 计算到所有剩余节点的最小
    left_corners = list(state[1])
    current_index = state[0]

    # 计算欧式距离可以保证小于?
    def calculate_distance(p):
        # return abs(p[0] - current_index[0]) + abs(p[1] - current_index[1])
        return ( (current_index[0] - p[0]) ** 2 + (current_index[1] - p[1]) ** 2 ) ** 0.5
    if len(left_corners) == 0:
        min_distance = 0
    else:
        min_distance = min(map(calculate_distance, left_corners))

    # 其它节点带来的偏移值
    if len(left_corners) == 4:
        offset = 2 * problem.s + problem.l
    elif len(left_corners) == 3:
        offset = problem.s + problem.l
    elif len(left_corners) == 2:
        offset = abs(left_corners[0][0] - left_corners[1][0]) + abs(left_corners[0][1] - left_corners[1][1])
    else:
        offset = 0


    return min_distance + offset # Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic_origin(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    # 计算所有food与离其最近的food的距离
    # len(food_list) >= 2
    base_list = []
    for food in food_list:
        min_distance = 99999
        already_minimum = False
        for other_food in food_list:
            if food == other_food:
                continue
            distance = abs(food[0] - other_food[0]) + abs(food[1] - other_food[1])
            # 没有比1更小的距离了
            if distance == 1:
                already_minimum = True
                break
            if distance < min_distance:
                min_distance = distance
        if already_minimum:
            # base += 1
            base_list.append(1)
        else:
            # base += min_distance
            base_list.append(min_distance)
    # 去除最大值
    base_list.sort()
    base_list = base_list[:-1]
    for bases in base_list:
        base += bases
    # len(food_list) == 1
    if len(food_list) == 1:
        base = 0
    min_distance = 99999
    for food in food_list:
        distance = abs(food[0] - position[0]) + abs(food[1] - position[1])
        if distance < min_distance:
            min_distance = distance
    # len(food_list) == 0
    if len(food_list) == 0:
        min_distance = 0
        base = 0
    position_distance = min_distance
    # print 'base:' + str(base) + ' min_distance:' + str(min_distance)
    # print 'state:' + str(state) + ' base:' + str(base) + ' min_distance:' + str(min_distance) + ' total:' + str(base + min_distance)
    return min_distance + base

def foodHeuristic_all_manhattan(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    # 计算所有food与离其最近的food的距离
    # len(food_list) >= 2
    base_list = []
    for food in food_list:
        min_distance = 99999
        already_minimum = False
        for other_food in food_list:
            if food == other_food:
                continue
            distance = abs(food[0] - other_food[0]) + abs(food[1] - other_food[1])
            # 没有比1更小的距离了
            if distance == 1:
                already_minimum = True
                break
            if distance < min_distance:
                min_distance = distance
        if already_minimum:
            # base += 1
            base_list.append(1)
        else:
            # base += min_distance
            base_list.append(min_distance)
    # 去除最大值
    base_list.sort()
    base_list = base_list[:-1]
    for bases in base_list:
        base += bases
    # len(food_list) == 1
    if len(food_list) == 1:
        base = 0
    min_distance = 99999
    for food in food_list:
        distance = abs(food[0] - position[0]) + abs(food[1] - position[1])
        if distance < min_distance:
            min_distance = distance
    # len(food_list) == 0
    if len(food_list) == 0:
        min_distance = 0
        base = 0
    position_distance = min_distance
    # print 'base:' + str(base) + ' min_distance:' + str(min_distance)
    # print 'state:' + str(state) + ' base:' + str(base) + ' min_distance:' + str(min_distance) + ' total:' + str(base + min_distance)
    return min_distance + base

def foodHeuristic_all_euclidean(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    # 计算所有food与离其最近的food的距离
    # len(food_list) >= 2
    base_list = []
    for food in food_list:
        min_distance = 99999
        already_minimum = False
        for other_food in food_list:
            if food == other_food:
                continue
            distance = ((food[0] - other_food[0]) ** 2 + (food[1] - other_food[1]) ** 2) ** 0.5
            # 没有比1更小的距离了
            if distance == 1:
                already_minimum = True
                break
            if distance < min_distance:
                min_distance = distance
        if already_minimum:
            # base += 1
            base_list.append(1)
        else:
            # base += min_distance
            base_list.append(min_distance)
    # 去除最大值
    base_list.sort()
    base_list = base_list[:-1]
    for bases in base_list:
        base += bases
    # len(food_list) == 1
    if len(food_list) == 1:
        base = 0
    min_distance = 99999
    for food in food_list:
        distance = ((food[0] - position[0]) ** 2 + (food[1] - position[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
    # len(food_list) == 0
    if len(food_list) == 0:
        min_distance = 0
        base = 0
    position_distance = min_distance
    # print 'base:' + str(base) + ' min_distance:' + str(min_distance)
    # print 'state:' + str(state) + ' base:' + str(base) + ' min_distance:' + str(min_distance) + ' total:' + str(base + min_distance)
    return min_distance + base

def foodHeuristic_longest_manhattan(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    # len(food_list) >= 2
    base_list = []
    food_A = (-1, -1)
    food_B = (-1, -1)
    longest_distance = -1
    for food in food_list:
        for other_food in food_list:
            if food == other_food:
                continue
            distance = abs(food[0] - other_food[0]) + abs(food[1] - other_food[1])
            if distance > longest_distance:
                longest_distance = distance
                food_A = food
                food_B = other_food
    if len(food_list) >= 2:
        distance_A = abs(position[0] - food_A[0]) + abs(position[1] - food_A[1])
        distance_B = abs(position[0] - food_B[0]) + abs(position[1] - food_B[1])
        return longest_distance + min(distance_A, distance_B)
    elif len(food_list) == 1:
        return abs(position[0] - food_list[0][0]) + abs(position[1] - food_list[0][1])
    else:
        return 0

def foodHeuristic_longest_euclidean(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    # len(food_list) >= 2
    base_list = []
    food_A = (-1, -1)
    food_B = (-1, -1)
    longest_distance = -1
    for food in food_list:
        for other_food in food_list:
            if food == other_food:
                continue
            distance = ((food[0] - other_food[0]) ** 2 + (food[1] - other_food[1]) ** 2) ** 0.5
            if distance > longest_distance:
                longest_distance = distance
                food_A = food
                food_B = other_food
    if len(food_list) >= 2:
        distance_A = ((position[0] - food_A[0]) ** 2 + (position[1] - food_A[1]) ** 2) ** 0.5
        distance_B = ((position[0] - food_B[0]) ** 2 + (position[1] - food_B[1]) ** 2) ** 0.5
        return longest_distance + min(distance_A, distance_B)
    elif len(food_list) == 1:
        return ((position[0] - food_list[0][0]) ** 2 + (position[1] - food_list[0][1]) ** 2) ** 0.5
    else:
        return 0


def foodHeuristic_longest_manhattan_wall(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h=height, w=width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            # 多出来的wall_offset加上本身１
                            more_walls = wall_offset + 1
                        else:
                            break
                        wall_offset = wall_offset + 1
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            # 要找到全局最大的值
            max_walls = 0
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                # 此时已经有一道墙了
                more_walls = 0
                if full and inside:
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[0], p2[0])
                    top = max(p1[0], p2[0])
                    while top + offset < w or buttom - offset >= 0:
                        if (top + offset >= w or walls[top + offset][x]) and (
                                buttom - offset < 0 or walls[buttom - offset][x]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
                    # return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                more_walls = 0
                if full and inside:
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[1], p2[1])
                    top = max(p1[1], p2[1])
                    while top + offset < h or buttom - offset >= 0:
                        if (top + offset >= h or walls[y][top + offset]) and (
                                buttom - offset < 0 or walls[y][buttom - offset]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
                    # return straight_distance + 2
            straight_distance = straight_distance + 2 * max_walls
        return straight_distance

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    # len(food_list) >= 2
    base_list = []
    food_A = (-1, -1)
    food_B = (-1, -1)
    longest_distance = -1
    for food in food_list:
        for other_food in food_list:
            if food == other_food:
                continue
            distance = distan(food, other_food)
            if distance > longest_distance:
                longest_distance = distance
                food_A = food
                food_B = other_food
    if len(food_list) >= 2:
        distance_A = distan(food_A, position)
        distance_B = distan(food_B, position)
        # if not problem.heuristicInfo.__contains__('info'):
        #     problem.heuristicInfo['info'] = 1
        # elif problem.heuristicInfo['info'] > 30:
        #     pass
        # else:
        #     problem.heuristicInfo['info'] = problem.heuristicInfo['info'] + 1
        #     print str(food_A) + ' ' + str(food_B) + ' ' + str(longest_distance) + ' ' + str(min(distance_A, distance_B)) + ' ' + str(position)
        return longest_distance + min(distance_A, distance_B)
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    else:
        return 0


def foodHeuristic_longest_maze(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    # if problem.heuristicInof.__contains__('walls'):
    #     walls = problem.heuristicInof['walls']
    # else:
    #     problem.heuristicInof['walls'] = problem.

    def distan(p1, p2):
        return mazeDistance(p1, p2, state)

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    # len(food_list) >= 2
    base_list = []
    food_A = (-1, -1)
    food_B = (-1, -1)
    longest_distance = -1
    for food in food_list:
        for other_food in food_list:
            if food == other_food:
                continue
            distance = distan(food, other_food)
            if distance > longest_distance:
                longest_distance = distance
                food_A = food
                food_B = other_food
    if len(food_list) >= 2:
        distance_A = distan(position, food_A)
        distance_B = distan(position, food_B)
        return longest_distance + min(distance_A, distance_B)
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    else:
        return 0


def foodHeuristic_longestAndOne_manhattan(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    def distan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    food_A = (-1, -1)
    food_B = (-1, -1)
    longest_distance = -1
    for food in food_list:
        for other_food in food_list:
            if food == other_food:
                continue
            distance = abs(food[0] - other_food[0]) + abs(food[1] - other_food[1])
            if distance > longest_distance:
                longest_distance = distance
                food_A = food
                food_B = other_food
    if len(food_list) >= 3:
        min_value = 999999999
        food_list.remove(food_A)
        food_list.remove(food_B)
        for left_food in food_list:
            # S-M-A AB
            value1 = distan(left_food, position) + distan(left_food, food_A) + longest_distance
            # SA A-M-B
            value2 = distan(position, food_A) + distan(food_A, left_food) + distan(left_food, food_B)
            # SA AB B-M
            value3 = distan(position, food_A) + longest_distance + distan(left_food, food_B)
            # S-M-B BA
            value4 = distan(position, left_food) + distan(left_food, food_B) + longest_distance
            # SB B-M-A
            value5 = distan(position, food_B) + distan(food_B, left_food) + distan(left_food, food_A)
            # SB BA A-M
            value6 = distan(position, food_B) + distan(food_B, food_A) + distan(food_A, left_food)

            min_v = min(value1, value2, value3, value4, value5, value6)
            if min_v < min_value:
                min_value = min_v
        return min_value
    elif len(food_list) == 2:
        distance_A = abs(position[0] - food_A[0]) + abs(position[1] - food_A[1])
        distance_B = abs(position[0] - food_B[0]) + abs(position[1] - food_B[1])
        return longest_distance + min(distance_A, distance_B)
    elif len(food_list) == 1:
        return ((position[0] - food_list[0][0]) ** 2 + (position[1] - food_list[0][1]) ** 2) ** 0.5
    else:
        return 0


def foodHeuristic_longestAndOne_manhattan_wall(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h = height, w = width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            more_walls = wall_offset + 1
                        else:
                            break
                        wall_offset = wall_offset + 1
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
        return straight_distance

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    food_A = (-1, -1)
    food_B = (-1, -1)
    longest_distance = -1
    for food in food_list:
        for other_food in food_list:
            if food == other_food:
                continue
            distance = abs(food[0] - other_food[0]) + abs(food[1] - other_food[1])
            if distance > longest_distance:
                longest_distance = distance
                food_A = food
                food_B = other_food
    if len(food_list) >= 3:
        min_value = 999999999
        food_list.remove(food_A)
        food_list.remove(food_B)
        for left_food in food_list:
            # S-M-A AB
            value1 = distan(left_food, position) + distan(left_food, food_A) + longest_distance
            # SA A-M-B
            value2 = distan(position, food_A) + distan(food_A, left_food) + distan(left_food, food_B)
            # SA AB B-M
            value3 = distan(position, food_A) + longest_distance + distan(left_food, food_B)
            # S-M-B BA
            value4 = distan(position, left_food) + distan(left_food, food_B) + longest_distance
            # SB B-M-A
            value5 = distan(position, food_B) + distan(food_B, left_food) + distan(left_food, food_A)
            # SB BA A-M
            value6 = distan(position, food_B) + distan(food_B, food_A) + distan(food_A, left_food)

            min_v = min(value1, value2, value3, value4, value5, value6)
            if min_v < min_value:
                min_value = min_v
        return min_value
    elif len(food_list) == 2:
        distance_A = abs(position[0] - food_A[0]) + abs(position[1] - food_A[1])
        distance_B = abs(position[0] - food_B[0]) + abs(position[1] - food_B[1])
        return longest_distance + min(distance_A, distance_B)
    elif len(food_list) == 1:
        return ((position[0] - food_list[0][0]) ** 2 + (position[1] - food_list[0][1]) ** 2) ** 0.5
    else:
        return 0


def foodHeuristic_longestTuple_manhattan(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    def distan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0] + food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        # 找到距离最大的两条边
        food_A = {(-1, -1), (-1, -1)}
        max_distance_A = -1
        food_B = {(-1, -1), (-1, -1)}
        max_distance_B = -1
        distance_set = set()
        for food in food_list:
            for other_food in food_list:
                if food == other_food:
                    continue
                current_distance = distan(food, other_food)
                foods = {food, other_food}
                set.add((current_distance, foods))
        distance_list = list(distance_set).sort()
        max_tuple = distance_list[:2]

        if set.intersection(max_tuple[0][1], max_tuple[1][1]) == 1:
            pass
        else:
            pass


def foodHeuristic_biggestTriangle_manhattan_wall(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h = height, w = width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            more_walls = wall_offset + 1
                        else:
                            break
                        wall_offset = wall_offset + 1
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
        return straight_distance

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        max_distance = -1
        for food in food_list:
            for other_food in food_list:
                if food == other_food:
                    continue
                d = distan(food, other_food)
                if d > max_distance:
                    food_A = food
                    food_B = other_food
                    max_distance = d
        food_list.remove(food_A)
        food_list.remove(food_B)
        # 找到到前两个点距离最远的第三个点
        food_C = (-1, -1)
        max_distance_C = -1
        for third_food in food_list:
            d = distan(third_food, food_B) + distan(third_food, food_A)
            if d > max_distance_C:
                max_distance_C = d
                food_C = third_food

        d_AB = max_distance
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC




        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestTriangle_euclidean(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    def distan(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        max_distance = -1
        for food in food_list:
            for other_food in food_list:
                if food == other_food:
                    continue
                d = distan(food, other_food)
                if d > max_distance:
                    food_A = food
                    food_B = other_food
                    max_distance = d
        food_list.remove(food_A)
        food_list.remove(food_B)
        # 找到到前两个点距离最远的第三个点
        food_C = (-1, -1)
        max_distance_C = -1
        for third_food in food_list:
            d = distan(third_food, food_B) + distan(third_food, food_A)
            if d > max_distance_C:
                max_distance_C = d
                food_C = third_food

        d_AB = max_distance
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestTriangle2_manhattan(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    def distan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    # if len({food[0], second_food[0], third_food[0]}) == 2 or len({food[1], second_food[1], third_food[1]}) == 2:
                    #     continue
                    # if len({food[0], second_food[0], third_food[0]}) == 2 and len({food[1], second_food[1], third_food[1]}) != 2:
                    #     continue
                    # if len({food[0], second_food[0], third_food[0]}) != 2 and len({food[1], second_food[1], third_food[1]}) == 2:
                    #     continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        # if len({food[0], second_food[0], third_food[0]}) < 3 or len({food[1], second_food[1], third_food[1]}) < 3:
        #     print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(state[0])
        # 如果已经没有合法的三角形
        # if max_distance == -1:
        #
        #     # print len(food_list)
        #     # 退化为直线
        #     food_A = (-1, -1)
        #     food_B = (-1, -1)
        #     max_distance = -1
        #     for food in food_list:
        #         for food_other in food_list:
        #             if food_other == food:
        #                 continue
        #             d = distan(food, food_other)
        #             if d > max_distance:
        #                 max_distance = d
        #                 food_A = food
        #                 food_B = food_other
        #     print str(food_A) + ' ' + str(food_B) + ' ' + ' ' + str(max_distance + min(distan(position, food_A), distan(position, food_B)))
        #     return max_distance + min(distan(position, food_A), distan(position, food_B))

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(min(value1, value2, value3, value4, value5, value6))
        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestTriangle2_euclidean(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    def distan(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestTriangle2_manhattan_wall(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    walls = problem.walls

    def distan(p1, p2):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        if p1[0] == p2[0]:
            for idx in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                if walls[p1[0]][idx]:
                    straight_distance = straight_distance + 2
        elif p1[1] == p2[1]:
            for idx in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                if walls[idx][p1[1]]:
                    straight_distance = straight_distance + 2
        return straight_distance
        # return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    # if len({food[0], second_food[0], third_food[0]}) == 2 or len({food[1], second_food[1], third_food[1]}) == 2:
                    #     continue
                    # if len({food[0], second_food[0], third_food[0]}) == 2 and len({food[1], second_food[1], third_food[1]}) != 2:
                    #     continue
                    # if len({food[0], second_food[0], third_food[0]}) != 2 and len({food[1], second_food[1], third_food[1]}) == 2:
                    #     continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        # if len({food[0], second_food[0], third_food[0]}) < 3 or len({food[1], second_food[1], third_food[1]}) < 3:
        #     print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(state[0])
        # 如果已经没有合法的三角形
        # if max_distance == -1:
        #
        #     # print len(food_list)
        #     # 退化为直线
        #     food_A = (-1, -1)
        #     food_B = (-1, -1)
        #     max_distance = -1
        #     for food in food_list:
        #         for food_other in food_list:
        #             if food_other == food:
        #                 continue
        #             d = distan(food, food_other)
        #             if d > max_distance:
        #                 max_distance = d
        #                 food_A = food
        #                 food_B = food_other
        #     print str(food_A) + ' ' + str(food_B) + ' ' + ' ' + str(max_distance + min(distan(position, food_A), distan(position, food_B)))
        #     return max_distance + min(distan(position, food_A), distan(position, food_B))

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(min(value1, value2, value3, value4, value5, value6))
        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestTriangle2_manhattan_wall2(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    walls = problem.walls

    def distan(p1, p2):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        if p1[0] == p2[0]:
            for idx in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                if walls[p1[0]][idx]:
                    straight_distance = straight_distance + 2
        elif p1[1] == p2[1]:
            for idx in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                if walls[idx][p1[1]]:
                    straight_distance = straight_distance + 2
        else:
            # 宽
            horizon = p1[0] - p2[0]
            # 高
            height = p1[1] - p2[1]
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    full = full and walls[y][x]
                if full:
                    return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    full = full and walls[y][x]
                if full:
                    return straight_distance + 2
        return straight_distance
        # return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    # if len({food[0], second_food[0], third_food[0]}) == 2 or len({food[1], second_food[1], third_food[1]}) == 2:
                    #     continue
                    # if len({food[0], second_food[0], third_food[0]}) == 2 and len({food[1], second_food[1], third_food[1]}) != 2:
                    #     continue
                    # if len({food[0], second_food[0], third_food[0]}) != 2 and len({food[1], second_food[1], third_food[1]}) == 2:
                    #     continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        # if len({food[0], second_food[0], third_food[0]}) < 3 or len({food[1], second_food[1], third_food[1]}) < 3:
        #     print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(state[0])
        # 如果已经没有合法的三角形
        # if max_distance == -1:
        #
        #     # print len(food_list)
        #     # 退化为直线
        #     food_A = (-1, -1)
        #     food_B = (-1, -1)
        #     max_distance = -1
        #     for food in food_list:
        #         for food_other in food_list:
        #             if food_other == food:
        #                 continue
        #             d = distan(food, food_other)
        #             if d > max_distance:
        #                 max_distance = d
        #                 food_A = food
        #                 food_B = food_other
        #     print str(food_A) + ' ' + str(food_B) + ' ' + ' ' + str(max_distance + min(distan(position, food_A), distan(position, food_B)))
        #     return max_distance + min(distan(position, food_A), distan(position, food_B))

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(min(value1, value2, value3, value4, value5, value6))
        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestTriangle2_manhattan_wall3(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"


    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h = height, w = width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            more_walls = wall_offset + 1
                        else:
                            break
                        wall_offset = wall_offset + 1
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
        return straight_distance
    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        if not problem.heuristicInfo.__contains__('info'):
            problem.heuristicInfo['info'] = 1
        elif problem.heuristicInfo['info'] > 30:
            pass
        else:
            problem.heuristicInfo['info'] = problem.heuristicInfo['info'] + 1
            print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(min(value1, value2, value3, value4, value5, value6)) + ' ' + str(position)

        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestTriangle2_manhattan_wall4(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"


    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h=height, w=width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            # 要找到全局最大的值
            if not problem.heuristicInfo.__contains__('log'):
                problem.heuristicInfo['log'] = True
            max_walls = 0
            if problem.heuristicInfo['log']:
                print str(p1) + ' ' + str(p2) + ' 10'
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                # 此时已经有一道墙了
                more_walls = 0
                if full and inside:
                    if problem.heuristicInfo['log']:
                        print '10-wall detected at x=' + str(x)
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[0], p2[0])
                    top = max(p1[0], p2[0])
                    while top + offset < w or buttom - offset >= 0:
                        if (top + offset >= w or walls[top + offset][x]) and (
                                buttom - offset < 0 or walls[buttom - offset][x]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if problem.heuristicInfo['log']:
                    print '10-get max more walls=' + str(more_walls)
                if more_walls > max_walls:
                    max_walls = more_walls
                    # return straight_distance + 2
            if problem.heuristicInfo['log']:
                print str(p1) + ' ' + str(p2) + ' 01'
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                more_walls = 0
                if full and inside:
                    if problem.heuristicInfo['log']:
                        print '01-wall detected at y=' + str(y)
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[1], p2[1])
                    top = max(p1[1], p2[1])
                    while top + offset < h or buttom - offset >= 0:
                        if (top + offset >= h or walls[y][top + offset]) and (
                                buttom - offset < 0 or walls[y][buttom - offset]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
                if problem.heuristicInfo['log']:
                    print '10-get max more walls=' + str(more_walls)
                    # return straight_distance + 2
            straight_distance = straight_distance + 2 * max_walls
            if problem.heuristicInfo['log']:
                problem.heuristicInfo['log'] = False
        return straight_distance
    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        if not problem.heuristicInfo.__contains__('info'):
            problem.heuristicInfo['info'] = 1
        elif problem.heuristicInfo['info'] > 30:
            pass
        else:
            problem.heuristicInfo['info'] = problem.heuristicInfo['info'] + 1
            print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(min(value1, value2, value3, value4, value5, value6)) + ' ' + str(position)

        return min(value1, value2, value3, value4, value5, value6)


def foodHeuristic_biggestQuadrangle_manhattan(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    def distan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) == 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        return min(value1, value2, value3, value4, value5, value6)
    elif len(food_list) >= 4:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        food_D = (-1, -1)
        max_distance = -1
        for food_first in food_list:
            for food_second in food_list:
                for food_third in food_list:
                    for food_forth in food_list:
                        if len({food_first, food_second, food_third, food_forth}) < 4:
                            continue
                        d12 = distan(food_first, food_second)
                        d13 = distan(food_first, food_third)
                        d14 = distan(food_first, food_forth)
                        d23 = distan(food_second, food_third)
                        d24 = distan(food_second, food_forth)
                        d34 = distan(food_third, food_forth)
                        d_total = d12 + d13 + d14 + d23 + d24 + d34
                        if d_total > max_distance:
                            max_distance = d_total
                            food_A = food_first
                            food_B = food_second
                            food_C = food_third
                            food_D = food_forth

        d_AB = distan(food_A, food_B)
        d_AC = distan(food_A, food_C)
        d_AD = distan(food_A, food_D)
        d_BC = distan(food_C, food_B)
        d_BD = distan(food_D, food_B)
        d_CD = distan(food_C, food_D)

        # ABCD BCDA CDAB DABC
        v1 = distan(position, food_A) + d_AB + d_BC + d_CD
        v2 = distan(position, food_B) + d_BC + d_CD + d_AD
        v3 = distan(position, food_C) + d_CD + d_AD + d_AB
        v4 = distan(position, food_D) + d_AD + d_AB + d_BC
        # ABDC BDCA DCAB CABD
        v5 = distan(position, food_A) + d_AB + d_BD + d_CD
        v6 = distan(position, food_B) + d_BD + d_CD + d_AC
        v7 = distan(position, food_C) + d_AC + d_AB + d_BD
        v8 = distan(position, food_D) + d_CD + d_AC + d_AB
        # ACBD CBDA BDAC DACB
        v9 = distan(position, food_A) + d_AC + d_BC + d_BD
        v10 = distan(position, food_B) + d_BD + d_AD + d_AC
        v11 = distan(position, food_C) + d_BC + d_BD + d_AD
        v12 = distan(position, food_D) + d_AD + d_AC + d_BC
        # ACDB CDBA DBAC BACD
        v13 = distan(position, food_A) + d_AC + d_CD + d_BD
        v14 = distan(position, food_B) + d_AB + d_AC + d_CD
        v15 = distan(position, food_C) + d_CD + d_BD + d_AB
        v16 = distan(position, food_D) + d_BD + d_AB + d_AC
        # ADBC DBCA BCAD CADB
        v17 = distan(position, food_A) + d_AD + d_BD + d_BC
        v18 = distan(position, food_B) + d_BC + d_AC + d_AD
        v19 = distan(position, food_C) + d_AC + d_AD + d_BD
        v20 = distan(position, food_D) + d_BD + d_BC + d_AC
        # ADCB DCBA CBAD BADC
        v21 = distan(position, food_A) + d_AD + d_CD + d_BC
        v22 = distan(position, food_B) + d_AB + d_AD + d_CD
        v23 = distan(position, food_C) + d_BC + d_AB + d_AD
        v24 = distan(position, food_D) + d_CD + d_BC + d_AB

        # print min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16)
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(food_D)
        # print str(v1) + ' ' + str(v2) + ' ' + str(v3) + ' ' + str(v4) + ' ' + str(v5) + ' ' + str(v6) + ' ' + str(
        #     v7) + ' ' + str(v8) + ' ' + str(v9) + ' ' + str(v10) + ' ' + str(v11) + ' ' + str(v12) + ' ' + str(
        #     v13) + ' ' + str(v14) + ' ' + str(v15) + ' ' + str(v16)

        return min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22,
                   v23, v24)


def foodHeuristic_biggestQuadrangle_manhattan_wall(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    walls = problem.walls

    def distan(p1, p2):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        if p1[0] == p2[0]:
            for idx in range(min(p1[1], p2[1]), max(p1[1], p2[1])):
                if walls[p1[0]][idx]:
                    straight_distance = straight_distance + 2
        elif p1[1] == p2[1]:
            for idx in range(min(p1[0], p2[0]), max(p1[0], p2[0])):
                if walls[idx][p1[1]]:
                    straight_distance = straight_distance + 2
        return straight_distance

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) == 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        return min(value1, value2, value3, value4, value5, value6)
    elif len(food_list) >= 4:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        food_D = (-1, -1)
        max_distance = -1
        for food_first in food_list:
            for food_second in food_list:
                for food_third in food_list:
                    for food_forth in food_list:
                        if len({food_first, food_second, food_third, food_forth}) < 4:
                            continue
                        d12 = distan(food_first, food_second)
                        d13 = distan(food_first, food_third)
                        d14 = distan(food_first, food_forth)
                        d23 = distan(food_second, food_third)
                        d24 = distan(food_second, food_forth)
                        d34 = distan(food_third, food_forth)
                        d_total = d12 + d13 + d14 + d23 + d24 + d34
                        if d_total > max_distance:
                            max_distance = d_total
                            food_A = food_first
                            food_B = food_second
                            food_C = food_third
                            food_D = food_forth

        d_AB = distan(food_A, food_B)
        d_AC = distan(food_A, food_C)
        d_AD = distan(food_A, food_D)
        d_BC = distan(food_C, food_B)
        d_BD = distan(food_D, food_B)
        d_CD = distan(food_C, food_D)

        # ABCD BCDA CDAB DABC
        v1 = distan(position, food_A) + d_AB + d_BC + d_CD
        v2 = distan(position, food_B) + d_BC + d_CD + d_AD
        v3 = distan(position, food_C) + d_CD + d_AD + d_AB
        v4 = distan(position, food_D) + d_AD + d_AB + d_BC
        # ABDC BDCA DCAB CABD
        v5 = distan(position, food_A) + d_AB + d_BD + d_CD
        v6 = distan(position, food_B) + d_BD + d_CD + d_AC
        v7 = distan(position, food_C) + d_AC + d_AB + d_BD
        v8 = distan(position, food_D) + d_CD + d_AC + d_AB
        # ACBD CBDA BDAC DACB
        v9 = distan(position, food_A) + d_AC + d_BC + d_BD
        v10 = distan(position, food_B) + d_BD + d_AD + d_AC
        v11 = distan(position, food_C) + d_BC + d_BD + d_AD
        v12 = distan(position, food_D) + d_AD + d_AC + d_BC
        # ACDB CDBA DBAC BACD
        v13 = distan(position, food_A) + d_AC + d_CD + d_BD
        v14 = distan(position, food_B) + d_AB + d_AC + d_CD
        v15 = distan(position, food_C) + d_CD + d_BD + d_AB
        v16 = distan(position, food_D) + d_BD + d_AB + d_AC
        # ADBC DBCA BCAD CADB
        v17 = distan(position, food_A) + d_AD + d_BD + d_BC
        v18 = distan(position, food_B) + d_BC + d_AC + d_AD
        v19 = distan(position, food_C) + d_AC + d_AD + d_BD
        v20 = distan(position, food_D) + d_BD + d_BC + d_AC
        # ADCB DCBA CBAD BADC
        v21 = distan(position, food_A) + d_AD + d_CD + d_BC
        v22 = distan(position, food_B) + d_AB + d_AD + d_CD
        v23 = distan(position, food_C) + d_BC + d_AB + d_AD
        v24 = distan(position, food_D) + d_CD + d_BC + d_AB

        # print min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16)
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(food_D)
        # print str(v1) + ' ' + str(v2) + ' ' + str(v3) + ' ' + str(v4) + ' ' + str(v5) + ' ' + str(v6) + ' ' + str(
        #     v7) + ' ' + str(v8) + ' ' + str(v9) + ' ' + str(v10) + ' ' + str(v11) + ' ' + str(v12) + ' ' + str(
        #     v13) + ' ' + str(v14) + ' ' + str(v15) + ' ' + str(v16)

        return min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24)



def foodHeuristic_biggestQuadrangle_manhattan_wall3(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h = height, w = width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            more_walls = wall_offset + 1
                        else:
                            break
                        wall_offset = wall_offset + 1
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                if full and inside:
                    return straight_distance + 2
        return straight_distance

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) == 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        return min(value1, value2, value3, value4, value5, value6)
    elif len(food_list) >= 4:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        food_D = (-1, -1)
        max_distance = -1
        for food_first in food_list:
            for food_second in food_list:
                for food_third in food_list:
                    for food_forth in food_list:
                        if len({food_first, food_second, food_third, food_forth}) < 4:
                            continue
                        d12 = distan(food_first, food_second)
                        d13 = distan(food_first, food_third)
                        d14 = distan(food_first, food_forth)
                        d23 = distan(food_second, food_third)
                        d24 = distan(food_second, food_forth)
                        d34 = distan(food_third, food_forth)
                        d_total = d12 + d13 + d14 + d23 + d24 + d34
                        if d_total > max_distance:
                            max_distance = d_total
                            food_A = food_first
                            food_B = food_second
                            food_C = food_third
                            food_D = food_forth

        d_AB = distan(food_A, food_B)
        d_AC = distan(food_A, food_C)
        d_AD = distan(food_A, food_D)
        d_BC = distan(food_C, food_B)
        d_BD = distan(food_D, food_B)
        d_CD = distan(food_C, food_D)

        # ABCD BCDA CDAB DABC
        v1 = distan(position, food_A) + d_AB + d_BC + d_CD
        v2 = distan(position, food_B) + d_BC + d_CD + d_AD
        v3 = distan(position, food_C) + d_CD + d_AD + d_AB
        v4 = distan(position, food_D) + d_AD + d_AB + d_BC
        # ABDC BDCA DCAB CABD
        v5 = distan(position, food_A) + d_AB + d_BD + d_CD
        v6 = distan(position, food_B) + d_BD + d_CD + d_AC
        v7 = distan(position, food_C) + d_AC + d_AB + d_BD
        v8 = distan(position, food_D) + d_CD + d_AC + d_AB
        # ACBD CBDA BDAC DACB
        v9 = distan(position, food_A) + d_AC + d_BC + d_BD
        v10 = distan(position, food_B) + d_BD + d_AD + d_AC
        v11 = distan(position, food_C) + d_BC + d_BD + d_AD
        v12 = distan(position, food_D) + d_AD + d_AC + d_BC
        # ACDB CDBA DBAC BACD
        v13 = distan(position, food_A) + d_AC + d_CD + d_BD
        v14 = distan(position, food_B) + d_AB + d_AC + d_CD
        v15 = distan(position, food_C) + d_CD + d_BD + d_AB
        v16 = distan(position, food_D) + d_BD + d_AB + d_AC
        # ADBC DBCA BCAD CADB
        v17 = distan(position, food_A) + d_AD + d_BD + d_BC
        v18 = distan(position, food_B) + d_BC + d_AC + d_AD
        v19 = distan(position, food_C) + d_AC + d_AD + d_BD
        v20 = distan(position, food_D) + d_BD + d_BC + d_AC
        # ADCB DCBA CBAD BADC
        v21 = distan(position, food_A) + d_AD + d_CD + d_BC
        v22 = distan(position, food_B) + d_AB + d_AD + d_CD
        v23 = distan(position, food_C) + d_BC + d_AB + d_AD
        v24 = distan(position, food_D) + d_CD + d_BC + d_AB

        # print min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16)
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(food_D)
        # print str(v1) + ' ' + str(v2) + ' ' + str(v3) + ' ' + str(v4) + ' ' + str(v5) + ' ' + str(v6) + ' ' + str(
        #     v7) + ' ' + str(v8) + ' ' + str(v9) + ' ' + str(v10) + ' ' + str(v11) + ' ' + str(v12) + ' ' + str(
        #     v13) + ' ' + str(v14) + ' ' + str(v15) + ' ' + str(v16)

        return min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24)



def foodHeuristic_biggestQuadrangle_manhattan_wall4(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h=height, w=width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            more_walls = wall_offset + 1
                        else:
                            break
                        wall_offset = wall_offset + 1
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            # 要找到全局最大的值
            max_walls = 0
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                # 此时已经有一道墙了
                more_walls = 0
                if full and inside:
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[0], p2[0])
                    top = max(p1[0], p2[0])
                    while top + offset < w or buttom - offset >= 0:
                        if (top + offset >= w or walls[top + offset][x]) and (
                                buttom - offset < 0 or walls[buttom - offset][x]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
                    # return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                more_walls = 0
                if full and inside:
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[1], p2[1])
                    top = max(p1[1], p2[1])
                    while top + offset < h or buttom - offset >= 0:
                        if (top + offset >= h or walls[y][top + offset]) and (
                                buttom - offset < 0 or walls[y][buttom - offset]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
                    # return straight_distance + 2
            straight_distance = straight_distance + 2 * max_walls
        return straight_distance

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) == 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        return min(value1, value2, value3, value4, value5, value6)
    elif len(food_list) >= 4:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        food_D = (-1, -1)
        max_distance = -1
        for food_first in food_list:
            for food_second in food_list:
                for food_third in food_list:
                    for food_forth in food_list:
                        if len({food_first, food_second, food_third, food_forth}) < 4:
                            continue
                        d12 = distan(food_first, food_second)
                        d13 = distan(food_first, food_third)
                        d14 = distan(food_first, food_forth)
                        d23 = distan(food_second, food_third)
                        d24 = distan(food_second, food_forth)
                        d34 = distan(food_third, food_forth)
                        d_total = d12 + d13 + d14 + d23 + d24 + d34
                        if d_total > max_distance:
                            max_distance = d_total
                            food_A = food_first
                            food_B = food_second
                            food_C = food_third
                            food_D = food_forth

        d_AB = distan(food_A, food_B)
        d_AC = distan(food_A, food_C)
        d_AD = distan(food_A, food_D)
        d_BC = distan(food_C, food_B)
        d_BD = distan(food_D, food_B)
        d_CD = distan(food_C, food_D)

        # ABCD BCDA CDAB DABC
        v1 = distan(position, food_A) + d_AB + d_BC + d_CD
        v2 = distan(position, food_B) + d_BC + d_CD + d_AD
        v3 = distan(position, food_C) + d_CD + d_AD + d_AB
        v4 = distan(position, food_D) + d_AD + d_AB + d_BC
        # ABDC BDCA DCAB CABD
        v5 = distan(position, food_A) + d_AB + d_BD + d_CD
        v6 = distan(position, food_B) + d_BD + d_CD + d_AC
        v7 = distan(position, food_C) + d_AC + d_AB + d_BD
        v8 = distan(position, food_D) + d_CD + d_AC + d_AB
        # ACBD CBDA BDAC DACB
        v9 = distan(position, food_A) + d_AC + d_BC + d_BD
        v10 = distan(position, food_B) + d_BD + d_AD + d_AC
        v11 = distan(position, food_C) + d_BC + d_BD + d_AD
        v12 = distan(position, food_D) + d_AD + d_AC + d_BC
        # ACDB CDBA DBAC BACD
        v13 = distan(position, food_A) + d_AC + d_CD + d_BD
        v14 = distan(position, food_B) + d_AB + d_AC + d_CD
        v15 = distan(position, food_C) + d_CD + d_BD + d_AB
        v16 = distan(position, food_D) + d_BD + d_AB + d_AC
        # ADBC DBCA BCAD CADB
        v17 = distan(position, food_A) + d_AD + d_BD + d_BC
        v18 = distan(position, food_B) + d_BC + d_AC + d_AD
        v19 = distan(position, food_C) + d_AC + d_AD + d_BD
        v20 = distan(position, food_D) + d_BD + d_BC + d_AC
        # ADCB DCBA CBAD BADC
        v21 = distan(position, food_A) + d_AD + d_CD + d_BC
        v22 = distan(position, food_B) + d_AB + d_AD + d_CD
        v23 = distan(position, food_C) + d_BC + d_AB + d_AD
        v24 = distan(position, food_D) + d_CD + d_BC + d_AB

        # print min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16)
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(food_D)
        # print str(v1) + ' ' + str(v2) + ' ' + str(v3) + ' ' + str(v4) + ' ' + str(v5) + ' ' + str(v6) + ' ' + str(
        #     v7) + ' ' + str(v8) + ' ' + str(v9) + ' ' + str(v10) + ' ' + str(v11) + ' ' + str(v12) + ' ' + str(
        #     v13) + ' ' + str(v14) + ' ' + str(v15) + ' ' + str(v16)

        return min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22,
                   v23, v24)


def foodHeuristic_biggestQuadrangle_euclidean(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"

    def distan(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.05

    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) == 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        return min(value1, value2, value3, value4, value5, value6)
    elif len(food_list) >= 4:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        food_D = (-1, -1)
        max_distance = -1
        for food_first in food_list:
            for food_second in food_list:
                for food_third in food_list:
                    for food_forth in food_list:
                        if len({food_first, food_second, food_third, food_forth}) < 4:
                            continue
                        d12 = distan(food_first, food_second)
                        d13 = distan(food_first, food_third)
                        d14 = distan(food_first, food_forth)
                        d23 = distan(food_second, food_third)
                        d24 = distan(food_second, food_forth)
                        d34 = distan(food_third, food_forth)
                        d_total = d12 + d13 + d14 + d23 + d24 + d34
                        if d_total > max_distance:
                            max_distance = d_total
                            food_A = food_first
                            food_B = food_second
                            food_C = food_third
                            food_D = food_forth

        d_AB = distan(food_A, food_B)
        d_AC = distan(food_A, food_C)
        d_AD = distan(food_A, food_D)
        d_BC = distan(food_C, food_B)
        d_BD = distan(food_D, food_B)
        d_CD = distan(food_C, food_D)

        # ABCD BCDA CDAB DABC
        v1 = distan(position, food_A) + d_AB + d_BC + d_CD
        v2 = distan(position, food_B) + d_BC + d_CD + d_AD
        v3 = distan(position, food_C) + d_CD + d_AD + d_AB
        v4 = distan(position, food_D) + d_AD + d_AB + d_BC
        # ABDC BDCA DCAB CABD
        v5 = distan(position, food_A) + d_AB + d_BD + d_CD
        v6 = distan(position, food_B) + d_BD + d_CD + d_AC
        v7 = distan(position, food_C) + d_AC + d_AB + d_BD
        v8 = distan(position, food_D) + d_CD + d_AC + d_AB
        # ACBD CBDA BDAC DACB
        v9 = distan(position, food_A) + d_AC + d_BC + d_BD
        v10 = distan(position, food_B) + d_BD + d_AD + d_AC
        v11 = distan(position, food_C) + d_BC + d_BD + d_AD
        v12 = distan(position, food_D) + d_AD + d_AC + d_BC
        # ACDB CDBA DBAC BACD
        v13 = distan(position, food_A) + d_AC + d_CD + d_BD
        v14 = distan(position, food_B) + d_AB + d_AC + d_CD
        v15 = distan(position, food_C) + d_CD + d_BD + d_AB
        v16 = distan(position, food_D) + d_BD + d_AB + d_AC
        # ADBC DBCA BCAD CADB
        v17 = distan(position, food_A) + d_AD + d_BD + d_BC
        v18 = distan(position, food_B) + d_BC + d_AC + d_AD
        v19 = distan(position, food_C) + d_AC + d_AD + d_BD
        v20 = distan(position, food_D) + d_BD + d_BC + d_AC
        # ADCB DCBA CBAD BADC
        v21 = distan(position, food_A) + d_AD + d_CD + d_BC
        v22 = distan(position, food_B) + d_AB + d_AD + d_CD
        v23 = distan(position, food_C) + d_BC + d_AB + d_AD
        v24 = distan(position, food_D) + d_CD + d_BC + d_AB

        # print min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16)
        # print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(food_D)
        # print str(v1) + ' ' + str(v2) + ' ' + str(v3) + ' ' + str(v4) + ' ' + str(v5) + ' ' + str(v6) + ' ' + str(
        #     v7) + ' ' + str(v8) + ' ' + str(v9) + ' ' + str(v10) + ' ' + str(v11) + ' ' + str(v12) + ' ' + str(
        #     v13) + ' ' + str(v14) + ' ' + str(v15) + ' ' + str(v16)

        return min(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22,
                   v23, v24)


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"


    if problem.heuristicInfo.__contains__('width'):
        width = problem.heuristicInfo['width']
        height = problem.heuristicInfo['height']
    else:
        bits = problem.walls.packBits()
        width = bits[0]
        height = bits[1]
        problem.heuristicInfo['width'] = width
        problem.heuristicInfo['height'] = height

    walls = problem.walls

    def distan(p1, p2, h=height, w=width):
        straight_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # 两点在同一行
        if p1[0] == p2[0]:
            max_walls = 0
            # 两点之间直线上的每一个格子
            for idx in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
                more_walls = 0
                # 直线上的格子是墙
                if walls[p1[0]][idx]:
                    wall_offset = 1
                    more_walls = 1
                    # 假定任意两个点之间一定有路径
                    while p1[0] - wall_offset >= 0 or p1[0] + wall_offset < w:
                        if (p1[0] - wall_offset < 0 or walls[p1[0] - wall_offset][idx]) and (
                                p1[0] + wall_offset >= w or walls[p1[0] + wall_offset][idx]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        elif p1[1] == p2[1]:
            max_walls = 0
            for idx in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
                more_walls = 0
                if walls[idx][p1[1]]:
                    wall_offset = 1
                    more_walls = 1
                    while p1[1] - wall_offset >= 0 or p1[1] + wall_offset < h:
                        if (p1[1] - wall_offset < 0 or walls[idx][p1[1] - wall_offset]) and (
                                p1[1] + wall_offset >= h or walls[idx][p1[1] + wall_offset]):
                            more_walls = wall_offset + 1
                            wall_offset = wall_offset + 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
            straight_distance = straight_distance + 2 * max_walls
        else:
            # 要找到全局最大的值
            max_walls = 0
            for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                full = True
                inside = False
                for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    inside = True
                    full = full and walls[y][x]
                # 此时已经有一道墙了
                more_walls = 0
                if full and inside:
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[0], p2[0])
                    top = max(p1[0], p2[0])
                    while top + offset < w or buttom - offset >= 0:
                        if (top + offset >= w or walls[top + offset][x]) and (
                                buttom - offset < 0 or walls[buttom - offset][x]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
                    # return straight_distance + 2
            for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                full = True
                inside = False
                for x in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    inside = True
                    full = full and walls[y][x]
                more_walls = 0
                if full and inside:
                    more_walls = 1
                    offset = 1
                    buttom = min(p1[1], p2[1])
                    top = max(p1[1], p2[1])
                    while top + offset < h or buttom - offset >= 0:
                        if (top + offset >= h or walls[y][top + offset]) and (
                                buttom - offset < 0 or walls[y][buttom - offset]):
                            more_walls = offset + 1
                            offset += 1
                        else:
                            break
                if more_walls > max_walls:
                    max_walls = more_walls
                    # return straight_distance + 2
            straight_distance = straight_distance + 2 * max_walls
        return straight_distance
    # 最简单的,对还存在的点进行计数
    # return foodGrid.count()
    food_list = foodGrid.asList()
    base = 0

    if len(food_list) == 0:
        return 0
    elif len(food_list) == 1:
        return distan(position, food_list[0])
    elif len(food_list) == 2:
        return distan(food_list[0], food_list[1]) + min(distan(position, food_list[0]), distan(position, food_list[1]))
    elif len(food_list) >= 3:
        food_A = (-1, -1)
        food_B = (-1, -1)
        food_C = (-1, -1)
        max_distance = -1
        for food in food_list:
            for second_food in food_list:
                for third_food in food_list:
                    if food == second_food or food == third_food or second_food == third_food:
                        continue
                    d = distan(food, second_food) + distan(second_food, third_food) + distan(food, third_food)
                    if d > max_distance:
                        food_A = food
                        food_B = second_food
                        food_C = third_food
                        max_distance = d

        d_AB = distan(food_A, food_B)
        d_BC = distan(food_C, food_B)
        d_AC = distan(food_C, food_A)

        # ABC order
        #   SABC
        value1 = distan(position, food_A) + d_AB + d_BC
        #   SBCA
        value2 = distan(position, food_B) + d_BC + d_AC
        #   SCAB
        value3 = distan(position, food_C) + d_AC + d_AB
        # ACB order
        #   SACB
        value4 = distan(position, food_A) + d_AC + d_BC
        #   SCBA
        value5 = distan(position, food_C) + d_BC + d_AB
        #   SBAC
        value6 = distan(position, food_B) + d_AB + d_AC

        # if not problem.heuristicInfo.__contains__('info'):
        #     problem.heuristicInfo['info'] = 1
        # elif problem.heuristicInfo['info'] > 30:
        #     pass
        # else:
        #     problem.heuristicInfo['info'] = problem.heuristicInfo['info'] + 1
        #     print str(food_A) + ' ' + str(food_B) + ' ' + str(food_C) + ' ' + str(min(value1, value2, value3, value4, value5, value6)) + ' ' + str(position)

        return min(value1, value2, value3, value4, value5, value6)



class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            # print nextPathSegment
            print currentState
            # print  nextPathSegment
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        current_path = search.breadthFirstSearch(problem)
        # return self.searchFunction(problem)
        return current_path
        # util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y]

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
