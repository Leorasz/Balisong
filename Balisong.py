import os
import json
import logging
import warnings
from openai import OpenAI
from transformers import pipeline

# from sklearn.metrics import roc_auc_score

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

warnings.filterwarnings("ignore")

"""
TODO: 
-Better validation
-Don't let punctuation mess it up
"""

logging.basicConfig(
    filename="error_log.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def logError(error):
    logging.error(error)
    logError(error)

class Balisong:

    def __init__(
        self,
        sentiment_threshhold=0.55,
        openai_model="gpt-4",
        final_validation=False,
        DEBUG=0,
    ):
        self.threshhold = sentiment_threshhold
        self.DEBUG = DEBUG
        self.client = OpenAI()
        self.final_validation = final_validation

        sentiment_model = (  # set up default sentiment classifier model
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        )
        if self.DEBUG >= 1:
            print("Getting the sentiment classifier model ready")
        self.model = pipeline("zero-shot-classification", model=sentiment_model)

        self.openai_model = openai_model

    class Node:  # node class for the causlang interpreter
        def __init__(self, name):
            self.name = name
            self.children = []
            self.color = "purple"

        def affect(self, loader):  # change colors of all children
            if self.color == "blue":
                for child in self.children:
                    child.color = "blue" if child.color != "red" else "red"
            loader += [child for child in self.children]

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
                f"Unsupported separation type, Causlang relationships can only by separated by a comma or newline"
            )

        for i in range(len(relationships)):  # remove any extra spaces
            if relationships[i][0] == " ":
                relationships[i] = relationships[i][1:]
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
            if ":" in relationship:
                components = relationship.split(":")
            else:
                if self.DEBUG >= 2:
                    print(f"Turning off {relationship}")
                if relationship[1:] in nodenames:
                    getNode(relationship[1:]).color = "red"
                else:
                    turnedOff = self.Node(relationship[1:])
                    nodenames.add(relationship[1:])
                    nodes.append(turnedOff)
                    turnedOff.color = "red"
                continue
            if components[0] not in nodenames:  # makes new node
                if self.DEBUG >= 2:
                    print(f"Causer {components[0]} not in nodenames, creating new node")
                causer = self.Node(components[0])
                nodenames.add(components[0])
                nodes.append(causer)

            else:
                causer = getNode(components[0])

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

            causer.children.append(affected)
            if self.DEBUG >= 2:
                print(f"Nodenames are now {nodenames}")

        if self.DEBUG >= 2:
            print("Node initialization complete, now onto changing colors")
            print(f"The children are {children}")
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

    def validateResult(self, system, ask, grading=True):
        verdict = self.getText(system, ask)
        if grading:  # this is for validating the causlang scripts
            verdict = self.findPayload(verdict, "\n")
            keyphrases = ["-inaccurate", "complete", "-incomplete"]
        else:  # this is for validating the final result
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

    def findPayload(
        self, stri, marker
    ):  # for finding the part of the stri after marker, useful when you want the model to walk through its thought process and also give a result to be used
        rekarm = marker[::-1]
        while stri[-1] == "\n":
            stri = stri[:-1]
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
        logError(f'"{marker}" was never found in the given string "{stri}"')

    def performCausalInference(self, text, scenario):  # do the whole thing
        if self.DEBUG >= 1:
            print("Performing causal inference")
        basicSystem = (
            "You are an expert in causality, ready to help the user with any request."
        )
        expertSystem = "You are an expert in a language called Causlang. Causlang was invented to make causal relationships computer readable, so then whether or not each entity is turned on can be calculated using causal reasoning. Causlang comes in 'relationships' that go 'causer:affected'. For example, if plants grow because of the sun, then the relationship would be 'sun shining:plants growing'. If bacteria died because the sun was shining, you can say 'sun shining:bacteria dying'. Another thing you can is do in manually turn an event off by saying '-event'. Make sure to only do this if you are explicitly told that the event is no longer happening. Tying this all together, if the text was 'The starship has several features. It uses its ventilation systems to clear out harmful pathogens. It has filtration systems to filter the water for the inhabitants and crew.' then the Causlang string for it would be 'starship online:ventilation systems online,starship online:filtration systems working,filtration systems working:clean water,ventilation systems online:pathogens being removed'. However, if you are also given the scenario 'The ventilation systems have broken', then the correct Causlang string would be 'starship online:ventilation systems online,starship online:filtration systems working,filtration systems working:clean water,ventilation systems online:pathogens being removed,-ventilation systems online'."
        if self.DEBUG >= 1:
            print("Starting out, making the original Causlang script.")
        while True:  # creating the initial causlang script
            initialGraph = self.getText(
                expertSystem,
                f"Make me the Causlang string for '{text}'. Walk through and explain each step of your reasoning. End your response with 'RES:' and then the all the relationships in Causlang in a comma-separated list.",
            )
            initialGraph = self.findPayload(initialGraph, "RES:")
            if self.DEBUG >= 1:
                print("Now validating the initialGraph")
            initialValidatorInput = f"A Causlang string has been generated for the following text: '{text}'. The Causlang string is '{initialGraph}'. Does this fit the scenario and accurately describe it causally? Walk through each step of your reasoning."
            if self.validateResult(expertSystem, initialValidatorInput):
                break

        if self.DEBUG >= 1:
            print("Now onto altering the graph to reflect the scenario.")
        while True:  # modifying the script to reflect the scenario
            scenarioGraph = self.getText(
                expertSystem,
                f"The original text was '{text}', and the Causlang generated for it was '{initialGraph}'. I want you to modify the Causlang string to reflect this scenario: '{scenario}'. Walk through each and every step of your reasoning. End your response with 'RES:' and then all the relationships in Causlang in a comma-separated list.",
            )
            scenarioGraph = self.findPayload(scenarioGraph, "RES:")
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
            if self.validateResult(expertSystem, scenarioValidatorInput):
                break

        if self.DEBUG >= 1:
            print(
                f"Now we have both graphs, time to get their results and then compare them. The initial graph is: \n{initialGraph}\nWhile the scneario graph is:\n{scenarioGraph}"
            )
        initialResults = self.interpretCauslang(initialGraph)  # computing the effects
        scenarioResults = self.interpretCauslang(scenarioGraph)
        if self.DEBUG >= 1:
            print("Results from both graphs calculated, now onto comparing them.")

        comparerInput = (  # now converting the difference into natural language
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
        comparerOutput = self.getText(basicSystem, comparerInput)
        comparisonValidatorInput = (
            text
            + " Now, "
            + scenario
            + ". Someone says that, given this, a reasonable outcome of the event is "
            + comparerOutput
            + ". Is this a reasonable thing that could have happened?"
        )
        # I've found that final result validation doesn't really work because it will always try to justify what has happened
        if not self.final_validation or self.validateResult(
            basicSystem, comparisonValidatorInput, grading=False
        ):
            return comparerOutput
        else:
            if self.DEBUG >= 1:
                print("Invalid output, starting over.")
            print("---------RESULT-----------")
            return self.performCausalInference(text, scenario)

    def makeModel(self):
        pass


bl = Balisong(DEBUG=2)
text = "At the Port of Los Angeles, about one-third of intermodal containers utilize the Port rail network, which includes one near-dock railyard and five on-dock railyards that serve the Port's seven container terminals. The use of on-dock rail is growing annually."
scenario = "If the Port's seven container slips were closed, how would that affect the port of los angeles?"
print(bl.performCausalInference(text, scenario))
