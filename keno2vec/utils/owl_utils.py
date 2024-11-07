from owlready2 import *
import os
import re
from tqdm import tqdm
import random

def get_paths_to_subclass_depth(begin, depth, num_paths):
    paths = []
    def dfs(begin, path, depth):
        if depth == 0:
            paths.append(path)
            return
        subclasses = list(begin.subclasses())
        if len(subclasses) == 0:
            paths.append(path)
            return
        for subclass in subclasses:
            dfs(subclass, path + [subclass], depth - 1)
    dfs(begin, [begin], depth)
    return random.sample(paths, num_paths)

def get_path_to_parent_depth(begin, depth, num_paths):
    paths = []
    def dfs(begin, path, depth):
        if depth == 0:
            paths.append(path)
            return
        parents = begin.is_a
        new_parents = []
        for p in parents:
            if isinstance(p, Restriction):
                continue
            if type(p) == Or:
                for c in p.Classes:
                    if isinstance(c, ThingClass):
                        new_parents.append(c)
            if isinstance(p, ThingClass):
                new_parents.append(p)
        if len(new_parents) == 0:
            paths.append(path)
            return
        for parent in new_parents:
            dfs(parent, path + [parent], depth - 1)
    dfs(begin, [begin], depth)
    return random.sample(paths, num_paths)

def get_path_to_subclass_breadth(begin, depth):
    path = []
    queue = []
    subclasses = list(begin.subclasses())
    if len(subclasses) == 0:
        return [begin]
    for subclass in subclasses:
        queue.append((subclass, 1, begin))
    while queue:
        current, current_depth, prev = queue.pop(0)
        if current_depth == depth + 1:
            break
        path.append(prev)
        path.append(current)
        subclasses = list(current.subclasses())
        for subclass in subclasses:
            queue.append((subclass, current_depth + 1, current))
    return path

def get_path_to_parent_breadth(begin, depth):
    path = []
    queue = []
    parents = begin.is_a
    new_parents = []
    for p in parents:
        if isinstance(p, Restriction):
            continue
        if type(p) == Or:
            for c in p.Classes:
                if isinstance(c, ThingClass):
                    new_parents.append(c)
        if isinstance(p, ThingClass):
            new_parents.append(p)
    if len(new_parents) == 0:
        return [begin]
    for parent in new_parents:
        queue.append((parent, 1, begin))
    while queue:
        current, current_depth, prev = queue.pop(0)
        if current_depth == depth + 1:
            break
        path.append(prev)
        path.append(current)
        parents = current.is_a
        new_parents = []
        for p in parents:
            if isinstance(p, Restriction):
                continue
            if type(p) == Or:
                for c in p.Classes:
                    if isinstance(c, ThingClass):
                        new_parents.append(c)
            if isinstance(p, ThingClass):
                new_parents.append(p)
        for parent in new_parents:
            queue.append((parent, current_depth + 1, current))
    return path
