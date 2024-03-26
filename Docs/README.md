# BALISONG DOCUMENTATION
The entire functionality of the code is in the Balisong class. Instantiate the class, and then get it to .performCausalInference(your_text, your_scenario).  
The Balisong class has some kwargs, which are:  
-openai_model: Choose a different openai model (gpt-4 by default)  
-exception_limit: How many exceptions you're okay with for one part of the code until you stop trying and raise an error (3 by default)  
There's also a "DEBUG" environment variable, whose levels are the following:  
0- Prints nothing (Default)  
1- Go through each major step   
2- More in detail on each step  
