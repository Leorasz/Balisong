# BALISONG DOCUMENTATION
The entire functionality of the code is in the CausalInferenceMaker class. Instantiate the class, and then get it to .performCausalInference(your_text, your_scenario).
The CausalInferenceMaker class has some inputs, which are:
-sentiment_threshhold: Manually set threshhold value that the validation's response has to get higher than
-sentiment_model: Choose a different sentiment classifier model (coming soon)
-openai_model: Choose a different openai mnodel
-DEBUG: Debug modes
    -0- Production- Prints nothing
    -1- Go through each major step
    -2- More in detail on each step
