import os
import re

OPENAI_API_KEY = "input key here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
torch.backends.cuda.sdp_kernel = "disable"

import wikipediaapi
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPTJForCausalLM

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

from datasets import load_dataset

dataset = load_dataset("gbharti/finance-alpaca")["train"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = 'initial generation.txt'

user_agent = "MyCoolTool/1.1 (https://example.org/MyCoolTool/; MyCoolTool@example.org) UsedBaseLibrary/1.4"

wiki_wiki = wikipediaapi.Wikipedia(user_agent, 'en')

page = wiki_wiki.page('Credit_card')

if not os.path.exists(file_path):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    llm = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16").to(device)

    question = []
    answer = []

    for content in dataset:
        question.append(content["instruction"])
        answer.append(content["output"])
        
    prompt = f"""
            Generate 3 insightful Q&A pairs about credit cards that are similar to these examples. The content should be based on {page.summary}.

            Question: {question[0]}
            Answer: {answer[0]}

            Question: {question[1]}
            Answer: {answer[1]}
            
            Question: {question[2]}
            Answer: {answer[2]}
            
            Question:
            """

        
    input_ids = tokenizer(prompt, return_tensors="pt", ).input_ids.to(device)
    generated_ids = llm.generate(
        input_ids,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        max_new_tokens=500,
        num_return_sequences=3,
        no_repeat_ngram_size=3
        )

    generated_text = tokenizer.decode(generated_ids[0])

    with open('initial generation.txt', 'w') as f:
        f.write(generated_text)

else:
    with open('initial generation.txt', 'r') as f:
        generated_text = f.read()

questions = re.findall(r'Question (.*?)\n', generated_text)
answers = re.findall(r'Answer (.*?)\n', generated_text)

hallucination = HallucinationMetric(model="gpt-3.5-turbo")

passed_questions = []
passed_answers = []
test_case = []

for i in range(len(questions)):
    print(questions[i], answers[i], type(page.summary))
    test_case.append(LLMTestCase(input=questions[i], actual_output=answers[i], context=[page.summary]))

test_case = EvaluationDataset(test_cases=test_case)

for i in test_case:
    score = hallucination.measure(i)
    print(f"score: {score}")
    if score > 0.4:
        passed_questions.append(i.input)
        passed_answers.append(i.actual_output)

print(f"\npassed_answers: {passed_answers}")
print(f"passed_questions: {passed_questions}\n")
        
#paraphrase

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

paraphrased_questions = []
paraphrased_answers = []

for i, question in enumerate(passed_questions):
    paraphrased_questions.append(question)
    paraphrased_questions += paraphrase(question)
    paraphrased_answers.append(passed_answers[i])
    paraphrased_answers += paraphrase(passed_answers[i])

df = pd.DataFrame({
    'Question': paraphrased_questions,
    'Answer': paraphrased_answers
})

output_file = 'credit_card_qa_pairs.csv'
df.to_csv(output_file, index=False)


