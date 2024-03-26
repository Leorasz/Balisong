import os
import json
import causlang
from openai import OpenAI
from utils import logError, cleanText

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

try:
    DEBUG = int(os.getenv("DEBUG"))
except Exception:
    DEBUG = 0
    os.environ["DEBUG"] = "0"

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
    ):
        self.openai_model = openai_model
        self.exception_limit = exception_limit
        self.client = OpenAI()

    def getText(self, system, inp):  # receive output from ChatGPT
        completion = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": inp},
            ],
        )
        if DEBUG >= 2:
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
                    payload = cleanText(payload)
                    if DEBUG >= 2:
                        print(f"The payload of the string was {payload}")
                    return payload
            else:
                lettersIn = 0
        logError(f'"{marker}" was never found in the given string "{stri}"')

    def makeInitialGraph(self, text, system):
        if DEBUG >= 1:
            print("Starting out, making the original Causlang script.")
        initialInput = f"Make me the Causlang string for '{text}'. Walk through and explain each step of your reasoning. End your response with 'RES:' and then the all the relationships in Causlang in a comma-separated list."
        exceptionTimes = 0
        while True:
            try:
                initialGraph = self.getText(system, initialInput)
                initialGraph = self.findPayload(initialGraph, "RES:")
                break
            except Exception as e:
                if DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")

        if DEBUG >= 1:
            print("Now correcting the initialGraph")
        initialValidatorInput = f"A Causlang string has been generated for the following text: '{text}'. The Causlang string is '{initialGraph}'. Does this fit the scenario and accurately describe it causally, going over every aspect? Walk through each step of your reasoning. If it's incorrect, then make corrections. End your response with 'RES: ' and then either the old Causlang string if it was correct or the corrected Causlang string."
        exceptionTimes = 0
        while True:
            try:
                initialGraph = self.getText(system, initialValidatorInput)
                initialGraph = self.findPayload(initialGraph, "RES:")
                break
            except Exception as e:
                if DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")
        return initialGraph

    def makeScenarioGraph(self, text, scenario, system, initialGraph):
        if DEBUG >= 1:
            print("Now onto altering the graph to reflect the scenario.")
        scenarioInput = f"The original text was '{text}', and the Causlang generated for it was '{initialGraph}'. I want you to modify the Causlang string to reflect this scenario: '{scenario}'. Walk through each and every step of your reasoning. End your response with 'RES:' and then all the relationships in Causlang in a comma-separated list."
        exceptionTimes = 0
        while True:
            try:
                scenarioGraph = self.getText(system, scenarioInput)
                scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                break
            except Exception as e:
                if DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")

        if DEBUG >= 1:
            print("Validating the scenario graph")
        scenarioValidatorInput = f'The original text was "{text}", and the original Causlang generated for it was "{initialGraph}". Then, the scenario "{scenario}" took place, and the new Causlang is "{scenarioGraph}". Does this fit the original text and the scenario, and accurately describe every aspect? Walk through each step of your reasoning. If it\'s incorrect, make corrections. End your response with "RES:" and then either the old Causlang if it was correct or the new corrected Causlang if it wasn\'t.'
        exceptionTimes = 0
        while True:
            try:
                scenarioGraph = self.getText(system, scenarioValidatorInput)
                scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                break
            except Exception as e:
                if DEBUG >= 1: print(f"Caught exception {e}, trying again")
                exceptionTimes += 1
                if exceptionTimes == self.exception_limit:
                    logError("Got an exception too many times for this one, force stopping")
        return scenarioGraph

    def makeComparison(self, text, scenario, system, initialGraph, scenarioGraph):
        if DEBUG >= 1:
            print(
                f"Now we have both graphs, time to get their results and then compare them. The initial graph is: \n{initialGraph}\nWhile the scneario graph is:\n{scenarioGraph}"
            )
        initialResults = causlang.interpretCauslang(initialGraph)  # computing the effects
        scenarioResults = causlang.interpretCauslang(scenarioGraph)
        if DEBUG>=2:
            print(f"Initial results are: {initialResults}")
            print(f"Scenario results are: {scenarioResults}")
        if DEBUG >= 1:
            print("Results from both graphs calculated, now onto comparing them.")
        comparerInput = f"Here's the situation: {text} Here's the status of all the entities in this scenario: {initialResults} Now, {scenario} The status of everything is now {scenarioResults}. How would you describe the changes that took place? What entities are now active or inactive? Don't extrapolate or use your own knowledge, just describe the situation using the information that has been given to you."  # now converting the difference into natural language
        comparerOutput = self.getText(system, comparerInput)
        return comparerOutput

    def performCausalInference(self, text, scenario):  # do the whole thing
        if DEBUG >= 1:
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
            if DEBUG>=1: print(f"Now onto scenario: {scenario}")
            while True: 
                try:
                    initialGraph = self.getText(system,initialInput)
                    initialGraph = self.findPayload(initialGraph, "RES:")
                    print("Done did the initial initial")
                    initialValidatorInput = f"A Causlang string has been generated for the following text: '{text}'. The Causlang string is '{initialGraph}'. Does this fit the scenario and accurately describe it causally, going over every aspect? Walk through each step of your reasoning. If it's incorrect, then make corrections. End your response with 'RES: ' and then either the old Causlang string if it was correct or the corrected Causlang string." 
                    initialGraph = self.getText(system, initialValidatorInput)
                    initialGraph = self.findPayload(initialGraph, "RES:")
                    print("Done did the initial validation")
                    stuff = {"text":text,"initialCausalGraph":initialGraph,"initialActivations":causlang.interpretCauslang(initialGraph),"scenario":scenario}
                    scenarioInput = f"The original text was '{text}', and the Causlang generated for it was '{initialGraph}'. I want you to modify the Causlang string to reflect this scenario: '{scenario}'. Walk through each and every step of your reasoning. End your response with 'RES:' and then all the relationships in Causlang in a comma-separated list."
                    scenarioGraph = self.getText(system, scenarioInput)
                    scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                    print("Done did the initial scenario")
                    scenarioValidatorInput = f'The original text was "{text}", and the original Causlang generated for it was "{initialGraph}". Then, the scenario "{scenario}" took place, and the new Causlang is "{scenarioGraph}". Does this fit the original text and the scenario, and accurately describe every aspect? Walk through each step of your reasoning. If it\'s incorrect, make corrections. End your response with "RES:" and then either the old Causlang if it was correct or the new corrected Causlang if it wasn\'t.'
                    scenarioGraph = self.getText(system, scenarioValidatorInput)
                    scenarioGraph = self.findPayload(scenarioGraph, "RES:")
                    print("Done did the scenario validation")
                    stuff["scenarioCausalGraph"] = scenarioGraph
                    stuff["scenarioActivations"] = causlang.interpretCauslang(scenarioGraph)
                    final.append(stuff)
                    break
                except Exception: pass
        with open("poladata.json","w") as file:
            json.dump(final, file, indent=4)
    


