import os
from flask import Flask, render_template, request, session
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Load document
loader = PyPDFLoader('results/merged.pdf')

# Load and split documents
documents = loader.load_and_split()

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vectorstore
vectorstore = Chroma.from_documents(documents, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Create a memory object
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Create a question answering chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, memory=memory)

# Define the prompt template
template = """
Hello there, Data Scientist! 👋
My name is DocQA your friendly data retriever 🐕

The relevant key performance indicators (KPIs) to share with you are: 

KPI 1: {kpi1_name}
Value: {kpi1_value}

KPI 2: {kpi2_name}
Value: {kpi2_value}

KPI 3: {kpi3_name}
Value: {kpi3_value}

In addition to these KPIs, here's a relevant quotes from the emedding: "{quote}"

Now, let's dive into some important insights based on these KPIs:

1. Insight 1: {insight1}
2. Insight 2: {insight2}
3. Insight 3: {insight3}

Remember, data is the new oil, and you're the one refining it!
"""

# Create a prompt template
prompt_template = PromptTemplate.from_template(template)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')  # replace with your own secret key

@app.route('/')
def home():
    # Initialize chat history when starting a new session
    session['chat_history'] = []
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    # Update chat history with user's question
    session['chat_history'].append({"role": "user", "content": question})
    # Run the question through your Langchain QA system here
    answer = qa.run({"question": question, "chat_history": session['chat_history']})
    # Update chat history with assistant's answer
    session['chat_history'].append({"role": "assistant", "content": answer['message']})  # assuming answer is a dict with a 'message' key
    return answer

if __name__ == '__main__':
    app.run(debug=True)
