from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain,SequentialChain

import os
f = open('C:\\Users\\dahmedsiddiqui\\Desktop\\OPEN_AI_KEY.txt')
os.environ['OPENAI_API_KEY'] = f.read()


def translate_and_summarize(email):

    # Create Model
    llm = ChatOpenAI()

    # CREATE A CHAIN THAT DOES THE FOLLOWING:

    # Detect Language
    template1 = "Return the language this email is written in:\n{email}.\nONLY return the language it was written in."
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain_1 = LLMChain(llm=llm,
                       prompt=prompt1,
                       output_key="language")

    # Translate from detected language to English
    template2 = "Translate this email from {language} to English. Here is the email:\n" + email
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain_2 = LLMChain(llm=llm,
                       prompt=prompt2,
                       output_key="translated_email")

    # Return English Summary AND the Translated Email
    template3 = "Create a short summary of this email:\n{translated_email}"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain_3 = LLMChain(llm=llm,
                       prompt=prompt3,
                       output_key="summary")

    seq_chain = SequentialChain(chains=[chain_1, chain_2, chain_3],
                                input_variables=['email'],
                                output_variables=['language', 'translated_email', 'summary'],
                                verbose=True)
    return seq_chain(email)


#Begin
spanish_email = open('spanish_customer_email.txt').read()
print(spanish_email)
result = translate_and_summarize(spanish_email)

print(result.keys())
print(result['language'])
print(result['translated_email'])
print(result['summary'])
