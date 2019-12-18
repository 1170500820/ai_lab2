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
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    # util.raiseNotDefined()

    log = start_logger('dfs_test2')
    # 初始化Closed表与Open表,并将初始状态压入Open栈中
    Closed = set()
    Open = util.Stack()
    Open.push(problem.getStartState())
    # 记录每一个节点的爸爸
    father = {problem.getStartState():('empty', None)}
    log.info('初始化')


    found_goal = False
    while not Open.isEmpty():
        # 从Open表中取出一个节点并扩展,然后将其放入Closed表中
        currrent_status = Open.pop()
        if currrent_status in Closed:
            continue
        next_steps = problem.getSuccessors(currrent_status)
        Closed.add(currrent_status)
        log.info('扩展节点:' + str(currrent_status))
        log.info('得到下一步节点:' + str(next_steps))

        # 判断扩展的节点中是否有目标节点
        for step in next_steps:
            # 确认该节点没有在以前访问过
            log.info('检查节点:' + str(step))
            if step[0] in Closed:
                log.info('已经访问过')
                continue
            Open.push(step[0])
            father[step[0]] = (currrent_status, step[1])
            log.info('存入Open表')
            if problem.isGoalState(step[0]):
                found_goal = True
                log.info('找到目标节点')
                break
        if found_goal:
            break
    if found_goal:
        log.info('回溯路径')
        directions = []
        current = Open.pop()
        log.info('当前节点:' + str(current))
        # 当前节点的父亲,以及从父亲到当前节点的方向
        father2son = father[current]
        log.info('父亲信息:' + str(father2son))
        while father2son[0] != 'empty':
            directions.insert(0, father2son[1])
            current = father2son[0]
            father2son = father[current]
            log.info('当前节点:' + str(current))
            log.info('父亲信息:' + str(father2son))
        return directions
    else:
        raise Exception('找不到')



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    log = start_logger('bfs_test')
    # 初始化Closed表与Open表,并将初始状态压入Open队列中
    Closed = set()
    Open = util.Queue()
    Open.push(problem.getStartState())
    # 记录每一个节点的爸爸
    father = {problem.getStartState(): ('empty', None)}
    log.info('初始化')


    found_goal = False
    target = None
    while not Open.isEmpty():
        # 从Open表中取出一个节点并扩展,然后将其放入Closed表中
        currrent_status = Open.pop()
        # TODO 在这里忘记检验,结果节点扩展次数直接加了1000倍
        if currrent_status in Closed:
            continue
        next_steps = problem.getSuccessors(currrent_status)
        Closed.add(currrent_status)
        log.info('扩展节点:' + str(currrent_status))
        log.info('得到下一步节点:' + str(next_steps))

        # 判断扩展的节点中是否有目标节点
        for step in next_steps:
            # 确认该节点没有在以前访问过
            log.info('检查节点:' + str(step))
            if step[0] in Closed:
                log.info('已经访问过')
                continue
            Open.push(step[0])
            father[step[0]] = (currrent_status, step[1])
            log.info('存入Open表')
            if problem.isGoalState(step[0]):
                found_goal = True
                target = step[0]
                log.info('找到目标节点')
                break
        if found_goal:
            break
    if found_goal:
        log.info('回溯路径')
        directions = []
        # 找到最后一个
        current = target
        log.info('当前节点:' + str(current))
        # 当前节点的父亲,以及从父亲到当前节点的方向
        father2son = father[current]
        log.info('父亲信息:' + str(father2son))
        while father2son[0] != 'empty':
            directions.insert(0, father2son[1])
            current = father2son[0]
            father2son = father[current]
            log.info('当前节点:' + str(current))
            log.info('父亲信息:' + str(father2son))
        return directions
    else:
        raise Exception('找不到')


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    log = start_logger('ucs_test')
    # 初始化Closed表与Open表,并将初始状态压入Open队列中
    Closed = set()
    Open = util.PriorityQueue()
    Open.push((problem.getStartState(), []), 0)
    # 记录每一个节点的爸爸
    # father = {problem.getStartState(): ('empty', None)}
    log.info('初始化')


    found_goal = False
    target = None
    while not Open.isEmpty():
        # 从Open表中取出一个节点并扩展,然后将其放入Closed表中
        currrent_status = Open.pop()
        if currrent_status[0] in Closed:
            continue
        next_steps = problem.getSuccessors(currrent_status[0])
        Closed.add(currrent_status[0])
        log.info('扩展节点:' + str(currrent_status))
        log.info('得到下一步节点:' + str(next_steps))

        # 判断扩展的节点中是否有目标节点
        for step in next_steps:
            # 确认该节点没有在以前访问过
            log.info('检查节点:' + str(step))
            if step[0] in Closed:
                log.info('已经访问过')
                continue
            Open.push((step[0], currrent_status[1] + [step[1]]), problem.getCostOfActions(currrent_status[1] + [step[1]]))
            # father[step[0]] = (currrent_status, step[1])
            log.info('存入Open表')
            if problem.isGoalState(step[0]):
                found_goal = True
                target = currrent_status[1] + [step[1]]
                log.info('找到目标节点')
                break
        if found_goal:
            break
    if found_goal:
        # log.info('回溯路径')
        # directions = []
        # # 找到最后一个
        # current = target
        # log.info('当前节点:' + str(current))
        # # 当前节点的父亲,以及从父亲到当前节点的方向
        # father2son = father[current]
        # log.info('父亲信息:' + str(father2son))
        # while father2son[0] != 'empty':
        #     directions.insert(0, father2son[1])
        #     current = father2son[0]
        #     father2son = father[current]
        #     log.info('当前节点:' + str(current))
        #     log.info('父亲信息:' + str(father2son))
        # return directions
        return target
    else:
        raise Exception('找不到')


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # TODO 这个实现少了一步:Open表和Closed表中的重复

    # log = start_logger('bfs_test')
    # # 初始化Closed表与Open表,并将初始状态压入Open队列中
    # Closed = set()
    # Open = util.PriorityQueue()
    # h = heuristic(problem.getStartState(), problem)
    # Open.push(problem.getStartState(), 0 + h)
    # # 记录每一个节点的爸爸
    # father = {problem.getStartState(): ('empty', None)}
    # log.info('初始化')
    #
    # found_goal = False
    # target = None
    # while not Open.isEmpty():
    #     # 从Open表中取出一个节点并扩展,然后将其放入Closed表中
    #     currrent_status = Open.pop()
    #     if currrent_status in Closed:
    #         continue
    #     next_steps = problem.getSuccessors(currrent_status)
    #     Closed.add(currrent_status)
    #     log.info('扩展节点:' + str(currrent_status))
    #     log.info('得到下一步节点:' + str(next_steps))
    #
    #     # 判断扩展的节点中是否有目标节点
    #     for step in next_steps:
    #         # 确认该节点没有在以前访问过
    #         log.info('检查节点:' + str(step))
    #         if step[0] in Closed:
    #             log.info('已经访问过')
    #             continue
    #         h = heuristic(step[0], problem)
    #         Open.push(step[0], step[2] + h)
    #         father[step[0]] = (currrent_status, step[1])
    #         log.info('存入Open表')
    #         if problem.isGoalState(step[0]):
    #             found_goal = True
    #             target = step[0]
    #             log.info('找到目标节点')
    #             break
    #     if found_goal:
    #         break
    # if found_goal:
    #     log.info('回溯路径')
    #     directions = []
    #     # 找到最后一个
    #     current = target
    #     log.info('当前节点:' + str(current))
    #     # 当前节点的父亲,以及从父亲到当前节点的方向
    #     father2son = father[current]
    #     log.info('父亲信息:' + str(father2son))
    #     while father2son[0] != 'empty':
    #         directions.insert(0, father2son[1])
    #         current = father2son[0]
    #         father2son = father[current]
    #         log.info('当前节点:' + str(current))
    #         log.info('父亲信息:' + str(father2son))
    #     return directions
    # else:
    #     raise Exception('找不到')
    log = start_logger('astar_test')
    # 初始化Closed表与Open表,并将初始状态压入Open队列中
    Closed = set()
    Open = util.PriorityQueue()
    h = heuristic(problem.getStartState(), problem)
    Open.push((problem.getStartState(), []), 0 + h)
    # 记录每一个节点的爸爸
    # father = {problem.getStartState(): ('empty', None)}
    log.info('初始化')


    found_goal = False
    target = None
    while not Open.isEmpty():
        # 从Open表中取出一个节点并扩展,然后将其放入Closed表中
        currrent_status = Open.pop()
        if currrent_status[0] in Closed:
            continue
        next_steps = problem.getSuccessors(currrent_status[0])
        Closed.add(currrent_status[0])
        log.info('扩展节点:' + str(currrent_status))
        log.info('得到下一步节点:' + str(next_steps))

        # 判断扩展的节点中是否有目标节点
        for step in next_steps:
            # 确认该节点没有在以前访问过
            log.info('检查节点:' + str(step))
            if step[0] in Closed:
                log.info('已经访问过')
                continue
            h = heuristic(step[0], problem)
            Open.push((step[0], currrent_status[1] + [step[1]]), h + problem.getCostOfActions(currrent_status[1] + [step[1]]))
            # father[step[0]] = (currrent_status, step[1])
            log.info('存入Open表')
            if problem.isGoalState(step[0]):
                found_goal = True
                target = currrent_status[1] + [step[1]]
                log.info('找到目标节点')
                break
        if found_goal:
            break
    if found_goal:
        # log.info('回溯路径')
        # directions = []
        # # 找到最后一个
        # current = target
        # log.info('当前节点:' + str(current))
        # # 当前节点的父亲,以及从父亲到当前节点的方向
        # father2son = father[current]
        # log.info('父亲信息:' + str(father2son))
        # while father2son[0] != 'empty':
        #     directions.insert(0, father2son[1])
        #     current = father2son[0]
        #     father2son = father[current]
        #     log.info('当前节点:' + str(current))
        #     log.info('父亲信息:' + str(father2son))
        # return directions
        return target
    else:
        raise Exception('找不到')


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
