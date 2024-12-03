from langchain_core.prompts import ChatPromptTemplate

prompt1 = """
Perform the following:
1. Read the content of the paper between the three backquotes and extract the attack method described in the paper, with a word limit of 100 words
2. Summarize the key features and steps of the attack method, with a word limit of 100 words.
3. Give the attack scenarios, the word limit is 100 words.
4. Provide the possible malicious prompt, word limit 100 words.
content:
```{content}```
Please answer in English.
"""

chat_prompt1 = ChatPromptTemplate.from_messages([
    ("system", prompt1),
    ("human", "{content}")
])

prompt2 = """
Perform the following:
1. Briefly summarize in one sentence, within 50 words
2. Infer the user's input intention, limited to 50 words
content:
```{content}```
Please answer in English.
"""

human_template = "{content}"
chat_prompt2 = ChatPromptTemplate.from_messages([
    ("system", prompt2),
    ("human", human_template),
])