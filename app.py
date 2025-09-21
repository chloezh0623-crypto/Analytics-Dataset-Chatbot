import streamlit as st


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pip', 'install langchain langchain-openai langchain-community openpyxl pandas python-dotenv faiss-cpu sentence-transformers')


# In[3]:


from langchain.chains import ConversationalRetrievalChain


# In[4]:


import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings  # For free embeddings if needed, but we'll use OpenAI
import difflib  # For suggesting similar names on not-found

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

llm = ChatOpenAI(model_name="gpt-5", temperature=0.0)


# In[5]:


# File path (same directory)
excel_path = "[Analytics] Dataset Library Input.xlsx"

# Load into DataFrame (update sheet_name if not default)
df = pd.read_excel(excel_path, sheet_name="Data flow")  # Or use engine='openpyxl' if issues

# Handle NaNs: Fill with empty strings for string columns
string_columns = ['Final table', 'Raw Data', 'Data Source', 'Data Format', 'Input Purpose', 
                  'Input cadence', 'Manual Update Required', 'Input Owner', 'DF or Not', 
                  'DF Name', 'DF Schedule', 'DF Purpose', 'DF Notes']
df[string_columns] = df[string_columns].fillna("")

# Extract unique entity names for exact matching and suggestions

df[string_columns] = df[string_columns].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
unique_final_tables = df['Final table'].unique().tolist()
unique_raw_data = df['Raw Data'].unique().tolist()
unique_df_names = df['DF Name'].unique().tolist()


print(f"Loaded {len(df)} rows. Unique Final tables: {len(unique_final_tables)}")
df.head()  # Display first few rows for verification


# In[6]:


# Initialize embeddings (using OpenAI)
embeddings = OpenAIEmbeddings()

# Function to create structured content from row
def row_to_content(row):
    content = (
        f"Final table: {row['Final table']}\n"
        f"Raw Data: {row['Raw Data']}\n"
        f"Data Source: {row['Data Source']}\n"
        f"Data Format: {row['Data Format']}\n"
        f"Input Purpose: {row['Input Purpose']}\n"
        f"Input cadence: {row['Input cadence']}\n"  # Use exact column name from your fix
        f"Manual Update Required: {row['Manual Update Required']}\n"
        f"Input Owner: {row['Input Owner']}\n"
        f"DF or Not: {row['DF or Not']}\n"
        f"DF Name: {row['DF Name']}\n"
        f"DF Schedule: {row['DF Schedule']}\n"
        f"DF Purpose: {row['DF Purpose']}\n"
        f"DF Notes: {row['DF Notes']}\n"
    )
    return content

# Create list of Documents
documents = []
for idx, row in df.iterrows():
    content = row_to_content(row)
    metadata = {
        'row_id': idx,
        'final_table': row['Final table'],
        'raw_data': row['Raw Data'],
        'df_name': row['DF Name'],
        'Manual_Update_Required': row['Manual Update Required'],
        'Input_Owner': row['Input Owner'],
        'DF_or_Not': row['DF or Not'],
        # Add other columns if needed for filtering
        'data_source': row['Data Source'],
        'input_purpose': row['Input Purpose'],
        'df_purpose': row['DF Purpose'],
    }
    documents.append(Document(page_content=content, metadata=metadata))

# Create vector store
vector_store = FAISS.from_documents(documents, embeddings)

print(f"Created vector store with {len(documents)} documents.")


# In[7]:


from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import re  # For generic name extraction if needed

# Updated entity prompt to also extract generic name if not categorized
entity_prompt = ChatPromptTemplate.from_template(
    "Extract entity names from the query. Categories: Final table, Raw Data, DF Name. "
    "Also extract any generic table/data name if not in categories (e.g., quoted or key term). "
    "Output as: Final table: name\nRaw Data: name\nDF Name: name\nGeneric name: name\nIf none, say 'None'.\nQuery: {query}"
)
entity_chain = entity_prompt | llm | StrOutputParser()

def extract_entities(query):
    response = entity_chain.invoke({"query": query})
    entities = {'final_table': None, 'raw_data': None, 'df_name': None, 'generic_name': None}
    for line in response.split('\n'):
        if 'Final table:' in line:
            entities['final_table'] = line.split(':', 1)[-1].strip()
        elif 'Raw Data:' in line:
            entities['raw_data'] = line.split(':', 1)[-1].strip()
        elif 'DF Name:' in line:
            entities['df_name'] = line.split(':', 1)[-1].strip()
        elif 'Generic name:' in line:
            entities['generic_name'] = line.split(':', 1)[-1].strip()
    return {k: v for k, v in entities.items() if v and v != 'None'}

def suggest_similar(name, unique_list, n=3):
    return difflib.get_close_matches(name, unique_list, n=n, cutoff=0.6)

def get_response(query):
    # Extract entities
    entities = extract_entities(query)
   
    # If no categorized entity, use generic and auto-detect type
    name = entities.get('generic_name') or entities.get('final_table') or entities.get('raw_data') or entities.get('df_name')
    if not name:
        return "No entity name detected in query. Please specify a table or data name."
   
    entity_type = None
    metadata_key = None
    if entities.get('final_table'):
        entity_type = 'final_table'
        metadata_key = 'final_table'
    elif entities.get('raw_data'):
        entity_type = 'raw_data'
        metadata_key = 'raw_data'
    elif entities.get('df_name'):
        entity_type = 'df_name'
        metadata_key = 'df_name'
    else:
        # Auto-detect: Check Final first
        if name in unique_final_tables:
            entity_type = 'final_table'
            metadata_key = 'final_table'
        elif name in unique_raw_data:
            entity_type = 'raw_data'
            metadata_key = 'raw_data'
        elif name in unique_df_names:
            entity_type = 'df_name'
            metadata_key = 'df_name'
   
    # Validate and suggest if no match
    if not entity_type:
        sim_final = suggest_similar(name, unique_final_tables)
        sim_raw = suggest_similar(name, unique_raw_data)
        sim_df = suggest_similar(name, unique_df_names)
        suggestions = []
        if sim_final:
            suggestions.append(f"Did you mean one of these Final tables: {', '.join(sim_final)}?")
        if sim_raw:
            suggestions.append(f"Did you mean one of these Raw Data: {', '.join(sim_raw)}?")
        if sim_df:
            suggestions.append(f"Did you mean one of these DF Names: {', '.join(sim_df)}?")
        return "\n".join(suggestions) if suggestions else f"No match or similar found for '{name}'. Please verify."
   
    filters = {metadata_key: name}
   
    # Manual retrieval to get ALL for final, with scores for debug
    if entity_type == 'final_table':
        docs_with_scores = vector_store.similarity_search_with_score(query, filter=filters, k=len(df))
        docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)  # Sort by relevance score descending
        docs = [doc for doc, score in docs_with_scores]  # Get all matching
    else:
        docs_with_scores = vector_store.similarity_search_with_score(query, filter=filters, k=10)
        docs = [doc for doc, score in docs_with_scores]
   
    # Debug: Count expected rows vs retrieved
    expected_count = len(df[df['Final table'] == name]) if entity_type == 'final_table' else len(df[df['Raw Data'] == name]) if entity_type == 'raw_data' else len(df[df['DF Name'] == name])
    retrieved_count = len(docs)
    #print(f"Expected rows for '{name}': {expected_count}. Retrieved: {retrieved_count}.")
    #if retrieved_count < expected_count:
    #    print("Warning: Not all rows retrieved—check name consistency in Excel (e.g., spaces).")
   
    # Fallback for token limits: If too many docs, take top 20 by score
    max_docs = 20  # Adjust based on testing; prevents 400 error
    if len(docs) > max_docs:
    #    print(f"Too many docs ({len(docs)}), truncating to top {max_docs} by relevance.")
        docs = docs[:max_docs]
   
    # Run chain with manual docs (override retriever)
    context = "\n\n".join(doc.page_content for doc in docs)
    # Fixed call: Use combine_documents_chain.llm_chain
    result = qa_chain.combine_documents_chain.llm_chain.predict(context=context, question=query)  # Direct call with context
    answer = result
   
    # Debug: List retrieved raw data for final
    if entity_type == 'final_table':
        retrieved_raw = set(doc.metadata.get('raw_data', '') for doc in docs)
    #    print(f"Retrieved Raw Data: {', '.join(retrieved_raw)}")
   
    return answer

print("Updated chatbot query function ready.")


# In[8]:


# Updated prompt template
prompt_template = """
You are a helpful assistant for querying dataset information from an Excel file. Use the following context (retrieved documents) to answer the question. No mentioning of the update and ownership related information unless specifically aksed. Provide your answer in a more natural way. Follow these rules strictly:

- Definitions:
  - Final table: Table applied directly to dashboard.
  - Raw Data: Data used to build Final table.
  - Data Source: Source of the Raw data.
  - Data Format: Format of the raw data.
  - Input Purpose: Purpose of the raw data (also part of the purpose for the final table).
  - Input cadence: How often the raw data is updated (impacts final table update).
  - Manual Update Required: If raw data needs manual update.
  - Input Owner: Person who owns the update and QA of the raw data.
  - DF or Not: If the final table uses a DF (data flow).
  - DF Name: Name of the DF used.
  - DF Schedule: How often the DF runs.
  - DF Purpose: Purpose of the DF (also the purpose of the final table).
  - DF Notes: Additional notes for DF (low priority).

- Relationships:
  - Final table is built from one or more Raw data.
  - If multiple Raw data, a DF blends them.
  - DF is always reliant on the raw data inputs—include raw level info first when relevant (e.g., purposes/frequencies from raw impact DF).
  - For questions on a Final table, search across ALL related Raw data and DF info, aggregating uniquely.

- Query Handling:
  - Use exact matches for names. If not found, suggest similar names and ask to verify.
  - If Raw data type: PRIORITIZE raw data level fields ONLY. Raw data level fields including Data Source, Data Format, Input Purpose, Input cadence, Manual Update Required, Input Owner (e.g., for purpose, use ONLY Input Purpose; ignore/deprioritize DF Purpose unless explicitly asked).
  - If DF type: use DF (Data Flow) level fields and raw data level fields. DF level fields including DF Name, DF Schedule, DF Purpose. emphasizing raw dependency to DF. When asked purpose of a Final table: First summarize ALL unique Input Purpose from raw data, which is the most important steps. then add DF Purpose if present. NO other unnecessary information such as ownership or any update related information.
  - If Final table type: Aggregate across ALL rows: Combine unique info from raw level fields and DF levels fields, emphasizing raw dependency.
  - When asked for the purpose of a Raw data: response with the "Input Purpose" and 'Final table' it's associated
  - Example: For purpose of a Final table: "First summarize ALL unique Input Purpose from raw data, which is the most important steps. then add [DF Purpose if present]. NO other unnecessary information such as ownership or any update related information."
  - Example: For refresh frequency/cadence of a Raw data: "Updated [Input cadence]; DF Schedule is secondary if applicable."
  - Example: For refresh frequency/cadence of a Final table: "Overall cadence depends on raw inputs: [list ALL unique Input cadence from matching raw data], blended via DF at [DF Schedule if present]."
  - Example: For source of a Final table: "Built from ALL these Raw Data: [list ALL unique Raw Data names and their Data Sources]."
  - Summarize uniquely without duplicates; use simple language for non-coders.
  - If no relevant info, suggest alternatives.

Context:
{context}

Question: {question}

Think step-by-step: Identify type (raw prioritizes raw level fields; final aggregates all raw + DF). Extract ALL relevant fields, aggregate/summarize per rules.

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Set up retriever (basic for now; can add filters later)
retriever = vector_store.as_retriever(search_kwargs={"k": 15})  # Retrieve top 15 docs

# Optional: Add compression for better context
from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# Update QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("QA chain updated with stricter prompt.")




# Streamlit app title and description
st.title("Analytics Dataset Library Chatbot")
st.write("Ask questions about the datasets used in Analytics Dashboards.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Your question (e.g., 'What datasets are available for sales analytics?')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            response = get_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


