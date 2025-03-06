from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os

# Set your OpenAI API key (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "api-key-pasted"

# Import necessary libraries for your RAG model
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the RAG model components
def initialize_rag_model():
    print("Initializing RAG model...")
    
    # Read the file
    with open('.\\ganj_data.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    ### FIX: Convert text into a list of `Document` objects ###
    documents = [Document(page_content=text)]  # Fixed here

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)  # Now `docs` is a list of Document objects

    # Create vector store
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_function)
    retriever = vectorstore.as_retriever()

    # Define prompt template
    
    prompt = PromptTemplate.from_template(
        """
توهان ھڪ مددگار آھيو جيڪو شاھ عبداللطيف ڀٽائي بابت معلومات ڏيڻ ۾ مدد ڪندو آھي. ھيٺ ڏنل حوالي (Context) مان صحيح معلومات حاصل ڪريو ۽ سوال جو جواب ڏيو.

مثال:
Q: شاھ عبداللطيف ڀٽائي جي ولادت ڪڏهن ٿي؟
Context: شاھ عبداللطيف ڀٽائيءَ جن جو جنم، سن 1689ع ۾ حيدرآباد ضلعي جي هاڻوڪي ٽنڊي آدم تعلقي جي هڪ ڳوٺ سُئي ڪنڌر، هالا حويلي ۾ ٿيو.
A: شاھ عبداللطيف ڀٽائيءَ جي ولادت 1689ع ۾ حيدرآباد ضلعي جي ڳوٺ سُئي ڪنڌر، هالا حويلي ۾ ٿي.

Q: شاھ عبداللطيف ڀٽائيءَ جو وفات ڪڏهن ٿي؟
Context: شاھ عبداللطيف ڀٽائيءَ جو وفات سن 1752ع ۾ 63 سالن جي عمر ۾ ٿي.
A: شاھ عبداللطيف ڀٽائيءَ جو وفات 22 ڊسمبر 1752ع ۾ 63 سالن جي عمر ۾ ٿي.

سوال: {question}
حوالو (Context): {context}

جواب:
"""
    )

    # Set up language model
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.3
    )

    # Format docs function
    def format_docs(docs_list):
        return "\n\n".join(doc.page_content for doc in docs_list)  # This was failing earlier because docs was a string

    # Create the RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG model initialized successfully!")
    return rag_chain

# Initialize the RAG model
try:
    rag_chain = initialize_rag_model()
    print("RAG model loaded successfully")
except Exception as e:
    print(f"Error initializing RAG model: {str(e)}")
    rag_chain = None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"response": "ڪجھ پڇڻ جي لاءِ مھرباني ڪري ڪجھ لکو!"})

    # Check for basic predefined responses first
    predefined_responses = {
        "السلام عليڪم": "وعليڪم السلام! اوهان ڪيئن آهيو؟ 😊",
        "توهان جو نالو ڇا آهي؟": "مان هڪ سنڌي چيٽ بوٽ آهيان.",
        "اوهان ڇا ڪري سگهو ٿا؟": "مان شاھ عبداللطيف ڀٽائي بابت معلومات ڏئي سگهان ٿو!",
        "خداحافظ": "الوداع! الله واهي 💙"
    }

    if user_message in predefined_responses:
        return jsonify({"response": predefined_responses[user_message]})

    # Use the RAG model for more complex queries
    if rag_chain:
        try:
            response = rag_chain.invoke(user_message)

            print("response", response)
            return jsonify({"response": response})
        
        except Exception as e:
            print(f"Error invoking RAG chain: {str(e)}")
            return jsonify({"response": f"معاف ڪجو، هڪ خرابي آئي آهي: {str(e)}"})
    else:
        return jsonify({"response": "موڊل لوڊ نه ٿي سگهيو"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
