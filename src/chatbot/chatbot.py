
import torch
import gradio as gr
from transformers import  pipeline 
from langchain.llms import HuggingFacePipeline 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer , AutoModelForCausalLM
import torch 
from peft import PeftModel
from transformers import GenerationConfig
import gradio as gr


tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5", 
    trust_remote_code=True, 
    torch_dtype=torch.float32
)

generation_config = GenerationConfig(
    max_length=150,
    # temperature=0.01,
    # top_p=0.95,
    # repetition_penalty=1.1,
    # do_sample=True,
    # use_cache=True,
    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    # transformers_version="4.33.1"
)

def generate_text(question):
    query = f'''Imagine you are a History Teacher teaching Indian Freedom Struggle. Answer the the following question.
        Question : {question}
        Answer:
    '''
    inputs = tokenizer(query,
                       return_tensors="pt",
                       return_attention_mask=False)
    outputs = model.generate(**inputs, generation_config=generation_config)
    text = tokenizer.batch_decode(outputs)[0]
    return text.split('\n')[3]


peft_model = PeftModel.from_pretrained(
    model, 
    "/trained/", 
    from_transformers=True
)
model = peft_model.merge_and_unload()
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=256,
#     temperature=0,
#     top_p=0.95,
#     repetition_penalty=1.2
# )
# local_llm = HuggingFacePipeline(pipeline=pipe)
# pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

# template = """
# As a history teacher teaching 10th grade students,
# answer the question in the context Indian Freedom struggle in 256 words.
# Question: {question}
# Answer:
# """

# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm_chain = LLMChain(prompt=prompt,
#                      llm=local_llm
#                      )
# question = "What is Satyagrapha?"
# print(llm_chain.run(question))


def echo(question, history):
#   answer = llm_chain.run(question)
  answer = generate_text(question)
  return answer

demo = gr.ChatInterface(fn=echo, examples=["What is Non-cooperation movement?", "What is Satyagraha?"], title="Education Chatbot")
demo.launch()