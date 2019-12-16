# coding=utf-8
# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import logging # TODO delete this

import util

# TODO delete this
def start_logger(logger_filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(logger_filename + ".txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    # util.raiseNotDefined()

    log = start_logger('dfs_test')

    visited = set()
    pathStack = util.Stack()
    directions = []

    current_status = problem.getStartState()
    pathStack.push(current_status)
    visited.add(current_status)

    is_target = problem.isGoalState(current_status)
    log.info('初始位置:' + str(current_status))
    log.info('当前栈' + str(pathStack.list))

    while not is_target:
        # 找到所有可用的下一步
        next_steps = problem.getSuccessors(current_status)
        next_steps.reverse()

        log.info('下一步:' + str(next_steps))

        found = False
        for step in next_steps:
            if step[0] in visited:
                continue
            else:
                found = True
                current_status = step[0]
                pathStack.push(current_status)
                visited.add(current_status)
                directions.append(step[1])
                is_target = problem.isGoalState(current_status)
                log.info('选中下一步:' + str(current_status))
                log.info('当前栈' + str(pathStack.list))
                break
        if found:
            continue
        # 未找到可用路径
        log.info('所有下一步均不可用,回退到上一步')
        if pathStack.isEmpty():
            raise Exception('寻路失败')
        directions = directions[:-1]
        pathStack.pop()
        current_status = pathStack.pop()
        pathStack.push(current_status)
        log.info('当前位置:' + str(current_status))
        log.info('当前栈' + str(pathStack.list))

    log.info('寻路完成')
    log.info('当前栈' + str(pathStack.list))
    log.info('历史路径:' + str(directions))
    return directions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    log = start_logger('bfs_test')

    visited = set()
    path_queue = util.Queue()
    directions = []
    last_step = {}

    current_status = problem.getStartState()
    start_status = current_status
    path_queue.push(current_status)
    visited.add(current_status)

    is_target = problem.isGoalState(current_status)
    log.info('初始位置:' + str(current_status))
    log.info('当前队列' + str(path_queue.list))


    while not is_target:
        current_status = path_queue.pop()
        log.info('选择节点:' + str(current_status))
        next_steps = problem.getSuccessors(current_status)
        log.info('下一步：' + str(next_steps))
        for step in next_steps:
            if step[0] in visited:
                continue
            else:
                is_target = problem.isGoalState(step[0])
                if is_target:
                    last_step[step[0]] = current_status
                    current_status = step[0]
                    break
                last_step[step[0]] = current_status
                visited.add(step[0])
                path_queue.push(step[0])
                log.info('添加节点:' + str(step[0]))

    log.info('找到路径:' + str(current_status))
    last = last_step[current_status]
    while last != start_status:
        next_steps = problem.getSuccessors(last)
        for step in next_steps:
            if step[0] == current_status:
                directions.insert(0, step[1])
                current_status = last
                last = last_step[last]
                break

    next_steps = problem.getSuccessors(start_status)
    for step in next_steps:
        if step[0] == current_status:
            directions.insert(0, step[1])
            break
    log.info('路径:' + str(directions))
    return directions

    return directions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
