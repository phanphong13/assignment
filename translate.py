from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


llm = OpenAI(temperature=0, openai_api_key = "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")

def translate_text(text, target_language): 
    prompt_template = "Translate the following text into {target_language}: {text}\n"
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    res = llm_chain.predict(text = text, target_language= target_language)
    
    return str(res)

def translate_json(json):
    if type(json["text"]) is str:
        text = json["text"]
        target_language = json["dest_language"] 
        translation = translate_text(text, target_language) 
        print(translation) 
    else:
        for i in json["text"]:
            text = i 
            target_language = json["dest_language"] 
            translation = translate_text(text, target_language) 
            print(translation) 



json_input = {
    "text": ["Hello", "I am Peter"],
    "dest_language": "vi"
}

translate_json(json_input)

