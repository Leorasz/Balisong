import os
import json
from openai import OpenAI
from transformers import pipeline
from sklearn.metrics import roc_auc_score

os.environ["OPENAI_API_KEY"] = "key"

"""
TODO: 
-Make it run
-Make updater method if different classification model
-Requirements
-README
-Docs
-Error handling
-Error logging
"""

class CausalInferenceMaker:

    def __init__(self, sentiment_threshhold=0.4, sentiment_model=None, openai_model="gpt-4", DEBUG=0):
        self.threshhold = sentiment_threshhold
        self.DEBUG = DEBUG
        self.client = OpenAI()

        if sentiment_model:
            raise ValueError("Functionality for other models not implemented yet")
            self.model = self.makeModel(sentiment_model)
        else:
            sentiment_model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            self.model = pipeline("zero-shot-classification",model=sentiment_model)

        self.openai_model = openai_model

    class Node:
        def __init__(self, name):
            self.name = name
            self.children = []
            if name[0]=="+": self.color = "blue"
            elif name[0]=="-": self.color = "red"
            else: self.color = "purple"

        def affect(self, loader):
            if self.color == "blue":
                for child, relationship in self.children:
                    if relationship == "+" and child.color == "purple": child.color = "blue"
                    elif relationship == "-": child.color = "red"
            loader+=[child for child, _ in self.children]

    def interpretCauslang(self, inp):
        nodenames = set()
        nodes = []
        children = set()

        if "," in inp: relationships = inp.split(",")
        elif "\n" in inp: relationships = inp.split("\n")
        else: raise ValueError(f"Unsupported separation type, Causlang relationships can only by separated by a comma or newline")

        for i in range(len(relationships)):
            if relationships[i][0] == " ": relationships[i] = relationships[i][1:]
        if self.DEBUG>=2: print(relationships)

        def getNode(name):
            if self.DEBUG>=2: print(f"Looking for node with name {name}")
            for node in nodes: 
                if node.name==name or node.name==name[1:] or node.name[1:]==name: return node
            raise ValueError(f"No node of name {name}")

        for relationship in relationships:
            if not relationship: continue
            if self.DEBUG>=2: print(f"On relationship {relationship}")
            components = relationship.split(":")
            assert len(components)==3, f"There were not 3 components- the relationship was {relationship}"
            if components[0] not in nodenames and components[0][1:] not in nodenames:
                if self.DEBUG>=2: print(f"Causer {components[0]} not in nodenames, creating new node")
                causer = self.Node(components[0])
                if components[0][0] == "+" or components[0][0] == "-": 
                    components[0] = components[0][1:]
                nodenames.add(components[0])
                nodes.append(causer)

            else: causer = getNode(components[0])

            if components[0][0] == "+" or components[0][0] == "-": 
                components[0] = components[0][1:]
            
            if components[1] not in nodenames:
                if self.DEBUG>=2: print(f"Affected {components[1]} not in nodenames, creating new node")
                nodenames.add(components[1])
                affected = self.Node(components[1])
                nodes.append(affected)
                children.add(components[1])
            else: affected = getNode(components[1])

            causer.children.append((affected, components[2]))
            if self.DEBUG>=2: print(f"Nodenames are now {nodenames}")
        
        if self.DEBUG>=2: print("Node initialization complete, now onto changing colors")
        layer = [getNode(name) for name in nodenames]
        while layer:
            newloader = []
            for i in layer:
                i.affect(newloader)
                layer = newloader

        if self.DEBUG>=2: print("Color changing complete, now converting to string")
        res = ""
        for node in nodes:
            activation = "active" if node.color == "blue" else "inactive"
            res += f"{node.name} is {activation}\n"
        
        return res

    def getText(self, system, inp):
        completion = self.client.chat.completions.create(
            model = self.openai_model,
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": inp}
            ]
        )
        if self.DEBUG>=2: print(f"Output of the OpenAI model was {completion.choices[0].message.content}")
        return completion.choices[0].message.content
    
    def validateResult(self, system, ask, grading=True):
        verdict = self.getText(system, ask)
        verdict = self.findPayload(verdict, "\n")
        if grading: keyphrases = ["-inaccurate","complete","-incomplete"]
        else: keyphrases = ["yes","-no","accurate","-inaccurate"]
        score = 0
        for key in keyphrases:
            if key[0] == "-":
                score += 1-self.model(verdict, key[1:])["scores"][0]
            else: 
                score += self.model(verdict, key)["scores"][0]
        score /= 3
        if self.DEBUG>=2: print(f"The score gotten by the result validation was {score}")
        return score > self.threshhold

    def findPayload(self, stri, marker):
        if self.DEBUG>=2: print(f"Finding the text after \"{marker}\" for the string \"{stri}\"")
        rekarm = marker[::-1]
        start = 0
        lettersIn = 0
        for i in range(len(stri)-1,-1,-1):
            if stri[i]==rekarm[lettersIn]:
                if lettersIn==0: start = i
                lettersIn += 1
                if lettersIn==len(rekarm):
                    payload = stri[start+1:]
                    payload = payload[1:] if payload[0] == "\"" else payload
                    payload = payload[:-1] if payload[-1] == "\"" else payload
                    if self.DEBUG>=2: print(f"The payload of the string was {payload}")
                    return payload
            else: lettersIn = 0
        raise ValueError(f"\"{marker}\" was never found in the given string \"{stri}\"")
    
    def performCausalInference(self, text, scenario):
        if self.DEBUG>=1: print("Performing causal inference")
        with open('long_strings.json', 'r') as f:
            strings = json.load(f)
        if self.DEBUG>=2: print("Finished loading long strings")

        if self.DEBUG>=1: print("Starting out, making the original Causlang script.")
        while True:
            initialGraph = self.getText(strings["basicSystem"], strings["initialGraphPreface"] + "The text I want you to do is \"" + text + "\"")
            initialGraph = self.findPayload(initialGraph, "RES: ") 
            initialValidatorInput = strings["initialValidatorPreface"] + " The original text was \"" + text + "\" and the Causlang string generated for it that you have to verify is \"" + initialGraph + "\". Walk through each step and talk about your thought process."
            if self.validateResult(strings["expertSystem"], initialValidatorInput): break

        if self.DEBUG>=1: print("Now onto altering the graph to reflect the scenario.")
        while True:
            scenarioGraph = self.getText(strings["expertSystem"], "The original text was \"" + text + "\", and the Causlang generated for it was \"" + initialGraph + "\". I want you to modify the Causlang to relfect this scenario: \"" + scenario + "\". Walk through each and every step of your reasoning. End your response with \"RES:\" and then all the relationships in Causlang in a comma-separated list.")
            scenarioGraph = self.findPayload(scenarioGraph, "RES: ")
            scenarioValidatorInput = "The original text was \"" + text + "\", and the original Causlang generated for it was \"" + initialGraph + "\". Then, the scenario \"" + scenario + "\" took place, and the new Causlang is \"" + scenarioGraph + "\". Does this fit the original text and the scenario? Walk through each step of your reasoning."
            if self.validateResult(strings["expertSystem"], scenarioValidatorInput): break
        
        if self.DEBUG>=1: print("Now we have both graphs, time to get their results and then compare them.")
        initialResults = self.interpretCauslang(initialGraph)
        scenarioResults = self.interpretCauslang(scenarioGraph)
        if self.DEBUG>=1: print("Results from both graphs calculated, now onto comparing them.")

        comparerInput = "Here's the situation:" + text + " Here's the status of all the entities in this scenario: " + initialResults + " Now, " + scenario + " The status of everything is now " + scenarioResults + ". How would you describe the changes that took place? What entities are now active or inactive?"
        comparerOutput = self.getText(strings["basicSystem"], comparerInput)
        comparisonValidatorInput = text + " Now, " + scenario + ". Someone says that, given this, a reasonable outcome of the event is " + comparerOutput + ". Is this a reasonable thing that could have happened?"
        return comparerOutput if self.validateResult(strings["basicSystem"], comparisonValidatorInput) else self.performCausalInference(text, scenario)


    def makeModel(self):
        pass
