import os
import json
from utils import logError, cleanText

try:
    DEBUG = int(os.getenv("DEBUG"))
except Exception:
    DEBUG = 0
    os.environ["DEBUG"] = "0"

"""
TODO:
-Optimize causlangToJSONEntity
"""

class Node:  # node class for the causlang interpreter
    def __init__(self, name):
        self.name = name
        self.children = []
        self.color = "purple"

    def affect(self, loader):  # change colors of all children
        if self.color == "blue":
            for child in self.children:
                child.color = "blue" if child.color != "red" else "red"
        elif self.color == "red":
            for child in self.children:
                child.color = "red"
        loader += self.children

def interpretCauslang(inp):
        nodenames = set()
        nodes = []
        children = set()

        if "," in inp:  # separate each relationship
            relationships = inp.split(",")
        elif "\n" in inp:
            relationships = inp.split("\n")
        else:
            logError(
                "Unsupported separation type, Causlang relationships can only by separated by a comma or newline"
            )

        if DEBUG >= 2:
            print(relationships)

        def getNode(
            name,
        ):  # for looking up the node object of a node given just its name
            for node in nodes:
                if node.name == name:
                    return node
            logError(f"No node of name {name}")

        for relationship in relationships:
            if not relationship:  # sometimes it gets empty relationships
                continue
            if DEBUG >= 2:
                print(f"On relationship {relationship}")
            if ":" not in relationship:
                relationship = cleanText(relationship)
                if relationship[0] != "-":
                    logError(f"Invalid relationship '{relationship}'")
                if relationship[1:] in nodenames:
                    getNode(relationship[1:]).color = "red"
                else:
                    turnedOff = Node(relationship[1:])
                    nodenames.add(relationship[1:])
                    nodes.append(turnedOff)
                    turnedOff.color = "red"
                continue
                
            components = relationship.split(":")
            components = [cleanText(component) for component in components]
            for component in components:
                if component[0] == "-":
                    logError("You can't start a relationship with a '-'!!")
                
            if components[0] not in nodenames:  # makes new node
                causer = Node(components[0])
                nodenames.add(components[0])
                nodes.append(causer)
            else:
                causer = getNode(components[0])

            if components[1] not in nodenames:
                nodenames.add(components[1])
                affected = Node(components[1])
                nodes.append(affected)
                children.add(components[1])
            else:
                affected = getNode(components[1])

            causer.children.append(affected)

        parents = list(nodenames - children)
        if DEBUG >= 2:
            print(f"The parents are {parents}")
        for i in parents:
            node = getNode(i)
            node.color = "blue" if node.color != "red" else "red"
        layer = [getNode(name) for name in nodenames]
        while layer:
            newloader = []
            for i in layer:
                i.affect(newloader)
                layer = newloader

        if DEBUG >= 2:
            print("Color changing complete, now converting to string")
        res = ""
        for node in nodes:
            activation = "active" if node.color == "blue" else "inactive"
            res += f"{node.name} is {activation}\n"

        return res

def causlangToJSONRelationship(causlang):
    relationships = causlang.split(",")
    final = []
    for r in relationships:
        if ":" in r:
            i = r.split(":")
            stuff = {"causer":i[0],"affected":i[1]}
        else:
            stuff = {"negated":r[1:]}
        final.append(stuff)
    with open("causlangrelationship.json","w") as file:
        json.dump(final,file,indent=4)

def causlangToJSONEntity(causlang):
    relationships = causlang.split(",")
    nodes = set()
    res = {}
    for relationship in relationships:
        if relationship[0] == "-":
            nodes.add(relationship[1:])
        else:
            components = relationship.split(":")
            nodes.add(components[0])
            nodes.add(components[1])
    for node in nodes:
        res[node] = {"negated":False,"active":True,"caused by":[],"affects":[]}
    for relationship in relationships:
        if relationship[0] == "-":
            res[relationship[1:]]["negated"] = True
        else:
            components = relationship.split(":")
            res[components[0]]["affects"].append(components[1])
            res[components[1]]["caused by"].append(components[0])
    activations = interpretCauslang(causlang)
    activations = activations.split("\n")
    for activation in activations:
        words = activation.split(" ")
        if words[-1] == "inactive":
            res[words[0]]["active"] = False

    with open("causlangentity.json","w") as file:
        json.dump(res, file, indent=4)
