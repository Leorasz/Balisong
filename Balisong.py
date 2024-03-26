import os
import json
import logging
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

logging.basicConfig(
    filename="error_log.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

def logError(error):
    logging.error(error)
    raise ValueError(error)

"""
TODO:
-Make interpretation of results more natural language
-Handle causal graphs that loop
"""

class Balisong:

    def __init__(
        self,
        openai_model="gpt-4",
        exception_limit = 3,
        DEBUG=0,
    ):
        self.openai_model = openai_model
        self.exception_limit = exception_limit
        self.DEBUG = DEBUG
        self.client = OpenAI()

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

    def cleanText(self, text):
        bads = [" ", "'", "\"", "\n", "."]
        while text[0] in bads:
            text = text[1:]
        while text[-1] in bads:
            text = text[:-1]
        return text
    
    def interpretCauslang(self, inp):
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

        if self.DEBUG >= 2:
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
            if self.DEBUG >= 2:
                print(f"On relationship {relationship}")
            if ":" not in relationship:
                relationship = self.cleanText(relationship)
                if relationship[0] != "-":
                    logError(f"Invalid relationship '{relationship}'")
                if relationship[1:] in nodenames:
                    getNode(relationship[1:]).color = "red"
                else:
                    turnedOff = self.Node(relationship[1:])
                    nodenames.add(relationship[1:])
                    nodes.append(turnedOff)
                    turnedOff.color = "red"
                continue
                
            components = relationship.split(":")
            components = [self.cleanText(component) for component in components]
            for component in components:
                if component[0] == "-":
                    logError("You can't start a relationship with a '-'!!")
                
            if components[0] not in nodenames:  # makes new node
                causer = self.Node(components[0])
                nodenames.add(components[0])
                nodes.append(causer)
            else:
                causer = getNode(components[0])

            if components[1] not in nodenames:
                nodenames.add(components[1])
                affected = self.Node(components[1])
                nodes.append(affected)
                children.add(components[1])
            else:
                affected = getNode(components[1])

            causer.children.append(affected)

        parents = list(nodenames - children)
        if self.DEBUG >= 2:
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

        if self.DEBUG >= 2:
            print("Color changing complete, now converting to string")
        res = ""
        for node in nodes:
            activation = "active" if node.color == "blue" else "inactive"
            res += f"{node.name} is {activation}\n"

        return res

    def getText(self, system, inp):  # receive output from ChatGPT
        completion = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": inp},
            ],
        )
        if self.DEBUG >= 2:
            print("----------------------------------")
            print(
                f"Output of the OpenAI model was {completion.choices[0].message.content}"
            )
            print("----------------------------------")
        return completion.choices[0].message.content

    def findPayload(
        self, stri, marker
    ):  # for finding the part of the stri after marker, useful when you want the model to walk through its thought process and also give a result to be used
        rekarm = marker[::-1]
        while stri[-1] == "\n":
            stri = stri[:-1]
        start = 0
        lettersIn = 0
        for i in range(len(stri) - 1, -1, -1):
            if stri[i].lower() == rekarm[lettersIn].lower():
                if lettersIn == 0:
                    start = i
                lettersIn += 1
                if lettersIn == len(rekarm):
                    payload = stri[start + 1 :]
                    payload = self.cleanText(payload)
                    if self.DEBUG >= 2:
                        print(f"The payload of the string was {payload}")
                    return payload
            else:
                lettersIn = 0
        logError(f'"{marker}" was never found in the given string "{stri}"')

    def makeInitialGraph(self, text, system):
        if self.DEBUG >= 1:
            print("Starting out, making the original Causlang script.")
        initialInput = f"Make me the Causlang string for '{text}'. Walk through and explain each step of your reasoning. End your response with 'RES:' and then the all the relationships in Causlang in a comma-separated list."
        exceptionTimes = 0
        while True:
            try:
                initialGraph = self.getText(system, initialInput)
                initialGraph = self.findPayload(initialGraph, "RES:")
                break
            except Exception as e:
                if self.DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")

        if self.DEBUG >= 1:
            print("Now correcting the initialGraph")
        initialValidatorInput = f"A Causlang string has been generated for the following text: '{text}'. The Causlang string is '{initialGraph}'. Does this fit the scenario and accurately describe it causally, going over every aspect? Walk through each step of your reasoning. If it's incorrect, then make corrections. End your response with 'RES: ' and then either the old Causlang string if it was correct or the corrected Causlang string."
        exceptionTimes = 0
        while True:
            try:
                initialGraph = self.getText(system, initialValidatorInput)
                initialGraph = self.findPayload(initialGraph, "RES:")
                break
            except Exception as e:
                if self.DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")
        return initialGraph

    def makeScenarioGraph(self, text, scenario, system, initialGraph):
        if self.DEBUG >= 1:
            print("Now onto altering the graph to reflect the scenario.")
        scenarioInput = f"The original text was '{text}', and the Causlang generated for it was '{initialGraph}'. I want you to modify the Causlang string to reflect this scenario: '{scenario}'. Walk through each and every step of your reasoning. End your response with 'RES:' and then all the relationships in Causlang in a comma-separated list."
        exceptionTimes = 0
        while True:
            try:
                scenarioGraph = self.getText(system, scenarioInput)
                scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                break
            except Exception as e:
                if self.DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")

        if self.DEBUG >= 1:
            print("Validating the scenario graph")
        scenarioValidatorInput = f'The original text was "{text}", and the original Causlang generated for it was "{initialGraph}". Then, the scenario "{scenario}" took place, and the new Causlang is "{scenarioGraph}". Does this fit the original text and the scenario, and accurately describe every aspect? Walk through each step of your reasoning. If it\'s incorrect, make corrections. End your response with "RES:" and then either the old Causlang if it was correct or the new corrected Causlang if it wasn\'t.'
        exceptionTimes = 0
        while True:
            try:
                scenarioGraph = self.getText(system, scenarioValidatorInput)
                scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                break
            except Exception as e:
                if self.DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")
        return scenarioGraph

    def makeComparison(self, text, scenario, system, initialGraph, scenarioGraph):
        if self.DEBUG >= 1:
            print(
                f"Now we have both graphs, time to get their results and then compare them. The initial graph is: \n{initialGraph}\nWhile the scneario graph is:\n{scenarioGraph}"
            )
        initialResults = self.interpretCauslang(initialGraph)  # computing the effects
        scenarioResults = self.interpretCauslang(scenarioGraph)
        if self.DEBUG>=2:
            print(f"Initial results are: {initialResults}")
            print(f"Scenario results are: {scenarioResults}")
        if self.DEBUG >= 1:
            print("Results from both graphs calculated, now onto comparing them.")
        comparerInput = f"Here's the situation: {text} Here's the status of all the entities in this scenario: {initialResults} Now, {scenario} The status of everything is now {scenarioResults}. How would you describe the changes that took place? What entities are now active or inactive? Don't extrapolate or use your own knowledge, just describe the situation using the information that has been given to you."  # now converting the difference into natural language
        comparerOutput = self.getText(system, comparerInput)
        return comparerOutput

    def performCausalInference(self, text, scenario):  # do the whole thing
        if self.DEBUG >= 1:
            print("Performing causal inference")
        basicSystem = (
            "You are an expert in causality, ready to help the user with any request."
        )
        with open("expertSystem.txt", "r") as file:
            expertSystem = file.read()
        initialGraph = self.makeInitialGraph(text, expertSystem)
        scenarioGraph = self.makeScenarioGraph(
            text, scenario, expertSystem, initialGraph
        )
        comparison = self.makeComparison(
            text, scenario, basicSystem, initialGraph, scenarioGraph
        )
        return comparison

    def makeData(self, text, scenarios):
        with open("expertSystem.txt","r") as file:
            system = file.read()
        initialInput = f"Make me the Causlang string for '{text}'. Walk through and explain each step of your reasoning. End your response with 'RES:' and then the all the relationships in Causlang in a comma-separated list."
        final = []
        for scenario in scenarios:
            print(f"Now onto scenario: {scenario}")
            while True: 
                try:
                    initialGraph = self.getText(system,initialInput)
                    initialGraph = self.findPayload(initialGraph, "RES:")
                    print("Done did the initial initial")
                    initialValidatorInput = f"A Causlang string has been generated for the following text: '{text}'. The Causlang string is '{initialGraph}'. Does this fit the scenario and accurately describe it causally, going over every aspect? Walk through each step of your reasoning. If it's incorrect, then make corrections. End your response with 'RES: ' and then either the old Causlang string if it was correct or the corrected Causlang string." 
                    initialGraph = self.getText(system, initialValidatorInput)
                    initialGraph = self.findPayload(initialGraph, "RES:")
                    print("Done did the initial validation")
                    stuff = {"text":text,"initialCausalGraph":initialGraph,"initialActivations":self.interpretCauslang(initialGraph),"scenario":scenario}
                    scenarioInput = f"The original text was '{text}', and the Causlang generated for it was '{initialGraph}'. I want you to modify the Causlang string to reflect this scenario: '{scenario}'. Walk through each and every step of your reasoning. End your response with 'RES:' and then all the relationships in Causlang in a comma-separated list."
                    scenarioGraph = self.getText(system, scenarioInput)
                    scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                    print("Done did the initial scenario")
                    scenarioValidatorInput = f'The original text was "{text}", and the original Causlang generated for it was "{initialGraph}". Then, the scenario "{scenario}" took place, and the new Causlang is "{scenarioGraph}". Does this fit the original text and the scenario, and accurately describe every aspect? Walk through each step of your reasoning. If it\'s incorrect, make corrections. End your response with "RES:" and then either the old Causlang if it was correct or the new corrected Causlang if it wasn\'t.'
                    scenarioGraph = self.getText(system, scenarioValidatorInput)
                    scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                    print("Done did the scenario validation")
                    stuff["scenarioCausalGraph"] = scenarioGraph
                    stuff["scenarioActivations"] = self.interpretCauslang(scenarioGraph)
                    final.append(stuff)
                    break
                except Exception as e:
                    print("Oh no got ", e)
        with open("poladata.json","w") as file:
            json.dump(final, file, indent=4)
    
    def causlangToJSONRelationship(self, causlang):
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

    def causlangToJSONEntity(self, causlang):
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
            res[node] = {"negated":False,"caused by":[],"affects":[]}
        for relationship in relationships:
            if relationship[0] == "-":
                res[relationship[1:]]["negated"] = True
            else:
                components = relationship.split(":")
                res[components[0]]["affects"].append(components[1])
                res[components[1]]["caused by"].append(components[0])
        with open("causlangentity.json","w") as file:
            json.dump(res, file, indent=4)




