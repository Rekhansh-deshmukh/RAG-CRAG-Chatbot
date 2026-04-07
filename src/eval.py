import pandas as pd
import re
from src.config import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_score(text):
    """Helper to find a number in LLM response."""
    match = re.search(r'\b([0-9]|10)\b', text)
    return int(match.group(1)) if match else 0

def run_comprehensive_eval(question: str, answer: str, contexts: list):
    """
    Evaluates RAG performance across 3 key metrics using LLM-as-a-Judge.
    """
    context_text = "\n\n".join([doc.page_content for doc in contexts]) if contexts else "No context."
    llm = get_llm(temperature=0)
    
    # 1. Faithfulness (Groundedness)
    faith_prompt = PromptTemplate(
        template="""Is the answer grounded ONLY in the context? 
Context: {context}
Answer: {answer}
Score 1-10 (10 = perfectly grounded, 1 = hallucinated). Output ONLY the number.
Score:""",
        input_variables=["context", "answer"]
    )
    
    # 2. Answer Relevancy
    rel_prompt = PromptTemplate(
        template="""Does the answer address the user question effectively?
Question: {question}
Answer: {answer}
Score 1-10 (10 = perfect, 1 = irrelevant). Output ONLY the number.
Score:""",
        input_variables=["question", "answer"]
    )

    # 3. Context Precision (Retriever Quality)
    prec_prompt = PromptTemplate(
        template="""Is the retrieved context actually useful for answering the question?
Question: {question}
Context: {context}
Score 1-10 (10 = highly useful, 1 = useless filler). Output ONLY the number.
Score:""",
        input_variables=["question", "context"]
    )

    try:
        faith_raw = (faith_prompt | llm | StrOutputParser()).invoke({"context": context_text, "answer": answer})
        rel_raw = (rel_prompt | llm | StrOutputParser()).invoke({"question": question, "answer": answer})
        prec_raw = (prec_prompt | llm | StrOutputParser()).invoke({"question": question, "context": context_text})
        
        faith_score = extract_score(faith_raw)
        rel_score = extract_score(rel_raw)
        prec_score = extract_score(prec_raw)
        
        avg_score = round((faith_score + rel_score + prec_score) / 3, 1)
        
        return {
            "Faithfulness": faith_score,
            "Answer Relevancy": rel_score,
            "Context Precision": prec_score,
            "Total RAG Score": avg_score
        }
    except Exception as e:
        return {"Error": str(e)}

def evaluate_batch(history_list):
    results = []
    for item in history_list:
        eval_data = run_comprehensive_eval(item['question'], item['answer'], item['context'])
        eval_data['Question'] = item['question'][:40] + "..."
        results.append(eval_data)
    
    return pd.DataFrame(results)
