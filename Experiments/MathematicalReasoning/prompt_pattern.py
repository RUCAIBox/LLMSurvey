PROMPT = {
    'ALPACA': "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n\
### Instruction:\n\
{}\n\n\
### Input:\n\
{}\n\n\
### Response:\n\
{}\n\n",

    'VANILLA': "Problem: {}\nSolution: Let's think step by step. {}\n",

    'FALCON': ">>QUESTION<<\n{}\n>>ANSWER<<\n{}\n",
}

STOP_WORD = {
    'ALPACA': "### Instruction:",
    'VANILLA': "Problem: ",
    'FALCON': ">>QUESTION<<",
}