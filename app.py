import os
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load Environment Variables
load_dotenv()

# Load LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Load Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Vector Database
vector_db = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Retriever
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Retrieval Grader
grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a retrieval grader.

Check if the retrieved document is relevant to the user's question.

Return ONLY:
yes
or
no

Do not explain.
"""
        ),

        (
            "human",
            """
Question:
{question}

Document:
{document}
"""
        )
    ]
)


retrieval_grader = grade_prompt | llm

# Query Rewriter
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query rewriter.

Rewrite the user question to improve document retrieval.

Return only the rewritten question.
"""
        ),

        (
            "human",
            "{question}"
        )
    ]
)


query_rewriter = rewrite_prompt | llm

# Corrective Retrieval
def corrective_retrieval(question):

    retrieved_docs = retriever.invoke(question)

    relevant_docs = []


    for doc in retrieved_docs:

        grade = retrieval_grader.invoke(
            {
                "question": question,
                "document": doc.page_content
            }
        )


        if "yes" in grade.content.lower():

            relevant_docs.append(doc)



    if relevant_docs:

        return relevant_docs



    rewritten_question = query_rewriter.invoke(
        {
            "question": question
        }
    )


    new_query = rewritten_question.content


    print(
        "Rewritten Query:",
        new_query
    )


    return retriever.invoke(new_query)


# Answer Generation
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer does not exist in the context, say:

I could not find the answer in the provided documents.

Do not use outside knowledge.
"""
        ),

        (
            "human",
            """
Question:

{question}


Context:

{context}
"""
        )
    ]
)



answer_chain = answer_prompt | llm



def generate_answer(question):

    if not question.strip():

        return (
            "Please enter a question.",
            ""
        )


    docs = corrective_retrieval(question)



    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )



    response = answer_chain.invoke(
        {
            "question": question,
            "context": context
        }
    )



    sources = "\n".join(
        list(
            set(
                [
                    doc.metadata["source"].split("/")[-1]
                    for doc in docs
                ]
            )
        )
    )



    return (
        response.content,
        sources
    )

# Interface
with gr.Blocks(
    title="Corrective RAG System"
) as demo:


    gr.Markdown(
        """
        # 📚 Corrective RAG System
        
        Ask questions from your course documents.
        
        The system evaluates retrieved context,
        corrects weak retrieval,
        and generates answers from verified documents only.
        """
    )



    question = gr.Textbox(
        label="Ask your question",
        placeholder="Example: What is Python?"
    )



    ask_btn = gr.Button(
        "Ask"
    )



    answer = gr.Textbox(
        label="Answer",
        lines=8
    )



    sources = gr.Textbox(
        label="Sources",
        lines=3
    )



    ask_btn.click(
        fn=generate_answer,
        inputs=question,
        outputs=[
            answer,
            sources
        ]
    )

# Run
if __name__ == "__main__":

    demo.launch()