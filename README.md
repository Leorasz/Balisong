# BALISONG
The aim of this project was to help LLMs deal with Causal Inference. For example, if I have “At the Port of Los Angeles, about one-third of intermodal containers utilize the Port rail network, which includes one near-dock railyard and five on-dock railyards that serve the Port's seven container terminals. The use of on-dock rail is growing annually,” and then I'm told "Presume the port rail network is offline. How would that affect the rest of the system?", then I should respond with something like "It would mean the Port's seven container terminals would not have rail network service. You would not be able to use the near-dock railyard, either." LLMs are notoriously bad at this, as they just predict the next token, and can't visualize the steps in between. So, I've developed a system to help it out. Here's how it works:  

I use a language I made called "Causlang", which is an easily parsable way to write out causal relationships so that their effects can be calculated.

I teach the LLM I'm using (in this case ChatGPT) how it works, and then tell it to convert the given text (i.e. "At the Port of Los Angeles...") into Causlang. I ask it again to check its work, and correct it if it's not right.

Then I ask it to change that Causlang script to reflect the scenario (i.e. "Presume the port rail network..."). The work is once again checked and corrected if need be.

Finally, I calculate whether or not each entity is activate for both before and after the scenario, feed the LLM this information, and ask it to describe the change. This is the end result.
