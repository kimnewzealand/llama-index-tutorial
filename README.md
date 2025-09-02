## LLama Index Tutorial

This is a tutorial for using LlamaIndex with HuggingFace.

>LlamaIndex is a complete toolkit for creating LLM-powered agents over your data using indexes and workflows. For this course we’ll focus on three main parts that help build agents in LlamaIndex: Components, Agents and Tools and Workflows.

## Benefits of LlamaIndex:

### **Core Features**
- **Purpose-built for RAG**: Specialized for Retrieval-Augmented Generation applications
- **100+ Data Connectors**: Built-in loaders for PDFs, databases, APIs, web pages, etc.
- **Advanced Indexing**: Vector stores, keyword indices, knowledge graphs, hybrid approaches
- **Smart Retrieval**: Multi-step reasoning, query decomposition, hierarchical search
- **Query Engines**: Built-in query optimization and routing capabilities

### **Production Ready**
- **Evaluation Framework**: Comprehensive metrics for RAG performance testing
- **Observability**: Built-in logging, tracing, and debugging tools
- **Intelligent Caching**: Cost and latency optimization strategies
- **Workflow Orchestration**: Visual workflow builder for complex processes
- **Enterprise Scale**: Built for large-scale data processing

### **Security & Reliability**
- **Mature Ecosystem**: Battle-tested components and established patterns
- **Input Sanitization**: Built-in protection against prompt injection
- **Access Control**: Role-based access to data sources
- **Audit Logging**: Comprehensive tracking of data access
- **Local Processing**: On-premises deployment options for sensitive data

### **Developer Experience**
- **High-level Abstractions**: Hides complexity while allowing customization
- **Rich Documentation**: Extensive guides and community support
- **Integration Friendly**: Works well with existing ML/AI stacks
- **Multi-modal Support**: Advanced handling of text, images, audio, video

### **Key Advantages over smolagents**
- **No Code Execution Risks**: Unlike smolagents' code generation approach
- **Specialized for Data**: Purpose-built vs general-purpose agent framework
- **Enterprise Features**: Production-ready vs experimental/prototype-focused
- **Proven Security**: Established security patterns vs newer, less tested approach

See tutorial https://huggingface.co/learn/agents-course/en/unit2/llama-index/introduction
Code has been updated to follow pythonic async/await pattern

## Setup

Follow these steps to set up the environment:

Use Python 3.13

1. **Create a virtual environment**:
    ```bash
    py -m venv .venv
    ```

2. **Activate the virtual environment**:
    - On Windows:
      ```bash
      .venv\Scripts\Activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt    
    ```

  ```bash
    HF_API_KEY="your_huggingface_api_key_here"
    ```
    
    **To get a HuggingFace API key:**
    1. Go to [HuggingFace](https://huggingface.co/)
    2. Create an account or sign in
    3. Go to your profile settings
    4. Navigate to "Access Tokens"
    5. Create a new token with "read" permissions
    6. Copy the token and paste it in your `.env` file

4. Create the tables:
    ```bash
    py tables.py
    ```

5. Run the agent:
    ```bash
    py main.py
    ```
  
## License

This project is for educational purposes only.
