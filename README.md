# BALISONG
The aim of this project was to help LLMs deal with Causal Inference. For example, if I have “At the Port of Los Angeles, about one-third of intermodal containers utilize the Port rail network, which includes one near-dock railyard and five on-dock railyards that serve the Port's seven container terminals. The use of on-dock rail is growing annually,” and then I'm told "Presume the port rail network is offline. How would that affect the rest of the system?", then I should respond with something like "It would mean the Port's seven container terminal would not have rail network service. You would not be able to use the near-dock railyard, either." LLMs are notoriously bad at this, as they just predict the next token, and can't visualize the steps in between. So, I've developed a system to help it out. Here's how it works:  

I created a sort of language I call "Causlang". Causlang is written as causer:affected. The effects of the string of causations are then calculated. For example, if a causes b, and b causes c, and then b is turned off, c gets turned off as well.  

I teach the LLM I'm using (in this case ChatGPT) how it works, and then tell it to convert the given text (i.e. "At the Port of Los Angeles...") into that language. I ask it again to check its work, and correct it if it's not right.

Then I ask it to change that Causlang script to reflect the scenario (i.e. "Presume the port rail network..."). The work is once again checked and corrected if need be.

Finally, I tell it whether or not each entity is activated for both before and after the scenario, and ask it to describe the change. This is the end result.
