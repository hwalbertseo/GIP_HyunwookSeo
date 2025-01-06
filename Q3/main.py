import os

OPENAI_API_KEY = "YOUR KEY HERE"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
torch.backends.cuda.sdp_kernel = "disable"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics.ragas import (
    RAGASAnswerRelevancyMetric,
    RAGASFaithfulnessMetric,
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric
)

from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


dataset = load_dataset("neural-bridge/rag-dataset-12000")["train"]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
unprocessed_documents = []
eval_dataset = []
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=300,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

for idx, content in enumerate(dataset):
    content_list = []
    if idx == 25:
        break
    
    unprocessed_documents.append(
        Document(
            page_content=content["context"]
        )
    )
    eval_dataset.append(
        LLMTestCase(
            input=content["question"],
            actual_output="",
            expected_output=content["answer"],
            retrieval_context=None
        ))

documents = []

for doc in unprocessed_documents:
    documents += text_splitter.split_documents([doc])


def create_rag_model(model_name, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cpu()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            device=-1  
        )
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    return rag_chain

rag_models = {
    "distilgpt2": create_rag_model("distilgpt2"),
    "facebook/opt-125m": create_rag_model("facebook/opt-125m"),
    "EleutherAI/gpt-neo-125M": create_rag_model("EleutherAI/gpt-neo-125M"),
}

eval_dataset = EvaluationDataset(test_cases=eval_dataset[:3])

# Metrics
deepeval = []
deepeval.append(ContextualPrecisionMetric(model="gpt-3.5-turbo"))
deepeval.append(ContextualRecallMetric(model="gpt-3.5-turbo"))
deepeval.append(AnswerRelevancyMetric(model="gpt-3.5-turbo"))
deepeval.append(FaithfulnessMetric(model="gpt-3.5-turbo"))

ragas = []
ragas.append(RAGASContextualPrecisionMetric(model="gpt-3.5-turbo"))
ragas.append(RAGASContextualRecallMetric(model="gpt-3.5-turbo"))
ragas.append(RAGASAnswerRelevancyMetric(model="gpt-3.5-turbo"))
ragas.append(RAGASFaithfulnessMetric(model="gpt-3.5-turbo"))

# Evaluate Models
with open("results_deepeval_cnn_dailymail.txt", "w") as f:
    for model_name, rag_model in rag_models.items():
        f.write(f"Evaluating {model_name}\n")
        
        for test_case in eval_dataset:
            try:
                actual_output = rag_model(test_case.input)
            except Exception as e:
                print(test_case.input)
            retrieval_context = [doc.page_content for doc in actual_output["source_documents"]]
            test_case.retrieval_context = retrieval_context
            test_case.actual_output = actual_output["result"]
            f.write(f"{test_case.input}\n")
            f.write(f"{test_case.actual_output}\n")
            f.write(f"{test_case.expected_output}\n")
            f.write(f"{test_case.retrieval_context}\n")
            f.write("-------------\n")
            
            # DEEPEVAL
            for metric in deepeval:
                metric.measure(test_case)
            
            # RAGAS
            for metric in ragas:
                metric.measure(test_case)
        
        f.write("DEEPEVAL\n")
        for metric in deepeval:
            f.write(f"{metric.score}\n")
            f.write(f"{metric.reason}\n")
        
        f.write("RAGAS\n")
        for metric in ragas:
            f.write(f"{metric.score}\n")
            f.write(f"{metric.reason}\n")
        f.write("============\n")
        f.write("============\n")

