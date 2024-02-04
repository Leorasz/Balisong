import os
import json
from openai import OpenAI
from transformers import pipeline

# from sklearn.metrics import roc_auc_score

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

"""
TODO: 
-Get rid of warning message
-Make updater method if different classification model
-Error logging
"""


class Balisong:

    def __init__(
        self,
        sentiment_threshhold=0.4,
        sentiment_model=None,
        openai_model="gpt-4",
        final_validation=False,
        DEBUG=0,
    ):
        self.threshhold = sentiment_threshhold
        self.DEBUG = DEBUG
        self.client = OpenAI()
        self.final_validation= final_validation

        if sentiment_model:
            raise ValueError("Functionality for other models not implemented yet")
            self.model = self.makeModel(sentiment_model)
        else:
            sentiment_model = ( #set up default sentiment classifier model
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            )
            if self.DEBUG >= 1:
                print("Getting the sentiment classifier model ready")
            self.model = pipeline("zero-shot-classification", model=sentiment_model)

        self.openai_model = openai_model

    class Node: #node class for the causlang interpreter
        def __init__(self, name):
            self.name = name
            self.children = []
            if name[0] == "+": #blue=on, red=off because inhibited, purple=off because nothing causing it
                self.color = "blue"
            elif name[0] == "-":
                self.color = "red"
            else:
                self.color = "purple"

        def affect(self, loader): #change colors of all children
            if self.color == "blue":
                for child, relationship in self.children:
                    if relationship == "+" and child.color == "purple":
                        child.color = "blue"
                    elif relationship == "-":
                        child.color = "red"
            loader += [child for child, _ in self.children]

    def interpretCauslang(self, inp):
        nodenames = set()
        nodes = []
        children = set()

        if "," in inp: #separate each relationship
            relationships = inp.split(",")
        elif "\n" in inp:
            relationships = inp.split("\n")
        else:
            raise ValueError(
                f"Unsupported separation type, Causlang relationships can only by separated by a comma or newline"
            )

        for i in range(len(relationships)): #remove any extra spaces
            if relationships[i][0] == " ":
                relationships[i] = relationships[i][1:]
        if self.DEBUG >= 2:
            print(relationships)

        def getNode(name): #for looking up the node object of a node given just its name
            if self.DEBUG >= 2:
                print(f"Looking for node with name {name}")
            for node in nodes:
                if node.name == name or node.name == name[1:] or node.name[1:] == name:
                    return node
            raise ValueError(f"No node of name {name}")

        for relationship in relationships:
            if not relationship: #sometimes it gets empty relationships
                continue
            if self.DEBUG >= 2:
                print(f"On relationship {relationship}")
            components = relationship.split(":")
            assert (
                len(components) == 3
            ), f"There were not 3 components- the relationship was {relationship}"
            if components[0] not in nodenames and components[0][1:] not in nodenames: #makes new node
                if self.DEBUG >= 2:
                    print(f"Causer {components[0]} not in nodenames, creating new node")
                causer = self.Node(components[0])
                if components[0][0] == "+" or components[0][0] == "-":
                    components[0] = components[0][1:]
                nodenames.add(components[0])
                nodes.append(causer)

            else:
                causer = getNode(components[0])

            if components[0][0] == "+" or components[0][0] == "-":
                components[0] = components[0][1:] #get rid of color setters to keep names consistent

            if components[1] not in nodenames:
                if self.DEBUG >= 2:
                    print(
                        f"Affected {components[1]} not in nodenames, creating new node"
                    )
                nodenames.add(components[1])
                affected = self.Node(components[1])
                nodes.append(affected)
                children.add(components[1])
            else:
                affected = getNode(components[1])

            causer.children.append((affected, components[2]))
            if self.DEBUG >= 2:
                print(f"Nodenames are now {nodenames}")

        if self.DEBUG >= 2:
            print("Node initialization complete, now onto changing colors")
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

    def getText(self, system, inp): #receive output from ChatGPT
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

    def validateResult(self, system, ask, grading=True):
        verdict = self.getText(system, ask)
        if grading: #this is for validating the causlang scripts
            verdict = self.findPayload(verdict, "\n")
            keyphrases = ["-inaccurate", "complete", "-incomplete"]
        else: #this is for validating the final result
            keyphrases = ["yes", "-no", "accurate", "-inaccurate"]
        score = 0
        for key in keyphrases:
            if key[0] == "-":
                score += 1 - self.model(verdict, key[1:])["scores"][0]
            else:
                score += self.model(verdict, key)["scores"][0]
        score /= len(keyphrases)
        if self.DEBUG >= 2:
            print(f"The score gotten by the result validation was {score}")
        if grading:
            return score > self.threshhold
        else:
            return score > 0.9

    def findPayload(self, stri, marker): #for finding the part of the stri after marker, useful when you want the model to walk through its thought process and also give a result to be used
        if self.DEBUG >= 2:
            print(f'Finding the text after "{marker}" for the string "{stri}"')
        rekarm = marker[::-1]
        start = 0
        lettersIn = 0
        for i in range(len(stri) - 1, -1, -1):
            if stri[i] == rekarm[lettersIn]:
                if lettersIn == 0:
                    start = i
                lettersIn += 1
                if lettersIn == len(rekarm):
                    payload = stri[start + 1 :]
                    payload = payload[1:] if payload[0] == '"' else payload
                    payload = payload[:-1] if payload[-1] == '"' else payload
                    if self.DEBUG >= 2:
                        print(f"The payload of the string was {payload}")
                    return payload
            else:
                lettersIn = 0
        raise ValueError(f'"{marker}" was never found in the given string "{stri}"')

    def performCausalInference(self, text, scenario): #do the whole thing
        if self.DEBUG >= 1:
            print("Performing causal inference")
        with open("long_strings.json", "r") as f: #get long strings, easier to store them in a different file
            strings = json.load(f)
        if self.DEBUG >= 2:
            print("Finished loading long strings")

        if self.DEBUG >= 1:
            print("Starting out, making the original Causlang script.")
        while True: #creating the initial causlang script
            initialGraph = self.getText(
                strings["basicSystem"],
                strings["initialGraphPreface"]
                + 'The text I want you to do is "'
                + text
                + '"',
            )
            initialGraph = self.findPayload(initialGraph, "RES: ")
            if self.DEBUG >= 1:
                print("Now validating the initialGraph")
            initialValidatorInput = (
                strings["initialValidatorPreface"]
                + ' The original text was "'
                + text
                + '" and the Causlang string generated for it that you have to verify is "'
                + initialGraph
                + '". Walk through each step and talk about your thought process.'
            )
            if self.validateResult(strings["expertSystem"], initialValidatorInput):
                break

        if self.DEBUG >= 1:
            print("Now onto altering the graph to reflect the scenario.")
        while True: #modifying the script to reflect the scenario
            scenarioGraph = self.getText(
                strings["expertSystem"],
                'The original text was "'
                + text
                + '", and the Causlang generated for it was "'
                + initialGraph
                + '". I want you to modify the Causlang to relfect this scenario: "'
                + scenario
                + '". Walk through each and every step of your reasoning. End your response with "RES:" and then all the relationships in Causlang in a comma-separated list.',
            )
            scenarioGraph = self.findPayload(scenarioGraph, "RES: ")
            if self.DEBUG >= 1:
                print("Validating the scenario graph")
            scenarioValidatorInput = (
                'The original text was "'
                + text
                + '", and the original Causlang generated for it was "'
                + initialGraph
                + '". Then, the scenario "'
                + scenario
                + '" took place, and the new Causlang is "'
                + scenarioGraph
                + '". Does this fit the original text and the scenario? Walk through each step of your reasoning.'
            )
            if self.validateResult(strings["expertSystem"], scenarioValidatorInput):
                break

        if self.DEBUG >= 1:
            print(
                "Now we have both graphs, time to get their results and then compare them."
            )
        initialResults = self.interpretCauslang(initialGraph) #computing the effects
        scenarioResults = self.interpretCauslang(scenarioGraph)
        if self.DEBUG >= 1:
            print("Results from both graphs calculated, now onto comparing them.")

        comparerInput = ( #now converting the difference into natural language
            "Here's the situation:"
            + text
            + " Here's the status of all the entities in this scenario: "
            + initialResults
            + " Now, "
            + scenario
            + " The status of everything is now "
            + scenarioResults
            + ". How would you describe the changes that took place? What entities are now active or inactive?"
        )
        comparerOutput = self.getText(strings["basicSystem"], comparerInput)
        comparisonValidatorInput = (
            text
            + " Now, "
            + scenario
            + ". Someone says that, given this, a reasonable outcome of the event is "
            + comparerOutput
            + ". Is this a reasonable thing that could have happened?"
        )
        #I've found that final result validation doesn't really work because it will always try to justify what has happened
        if not self.final_validation or self.validateResult(strings["basicSystem"], comparisonValidatorInput, grading=False):
            return comparerOutput
        else:
            if self.DEBUG >= 1:
                print("Invalid output, starting over.")
            print("---------RESULT-----------")
            return self.performCausalInference(text, scenario)
        
    def testLast(self, text, scenario):
        initialResults = "Port of Los Angeles online is active\nPort rail network online is active\nIntermodel containers online is active\nNear-dock railyard online is active\nOn dock railyard online is active\nContainer terminals online is active"
        scenarioResults = "Port of Los Angeles online is inactive\nPort rail network online is active\nIntermodel containers online is active\nNear-dock railyard online is inactive\nOn dock railyard online active\nContainer terminals online is inactive"
        scenarioResults = "Port of Los Angeles online is active\nPort rail network online is inactive\nIntermodel containers online is inactive\nNear-dock railyard online is inactive\nOn dock railyard online is active\nContainer terminals online is inactive"
        comparerInput = (
            "Here's the situation:"
            + text
            + " Here's the status of all the entities in this scenario: "
            + initialResults
            + " Now, "
            + scenario
            + " The status of everything is now "
            + scenarioResults
            + ". How would you describe the changes that took place? What entities are now active or inactive?"
        )
        comparerOutput = self.getText("You are an expert in causality, ready to help the user with anything they may ask.", comparerInput)
        comparisonValidatorInput = (
            text
            + " Now, "
            + scenario
            + ". Someone says that, given this, a reasonable outcome of the event is "
            + comparerOutput
            + ". Is this a reasonable thing that could have happened?"
        )
        print(self.validateResult("You are an expert in causality, ready to help the user with anything they may ask.",comparisonValidatorInput,grading=False))

    def makeModel(self):
        pass


bl = Balisong(DEBUG=2)
text = "At the Port of Los Angeles, about one-third of intermodal containers utilize the Port rail network, which includes one near-dock railyard and five on-dock railyards that serve the Port's seven container terminals. The use of on-dock rail is growing annually."
scenario = "Presume the port rail network is offline. How would that affect the rest of the system?"
print(bl.performCausalInference(text, scenario))
# bl.testLast(text, scenario)
