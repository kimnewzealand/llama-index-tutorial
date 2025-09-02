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

LlamaIndex supports three main types of reasoning agents:

1. Function Calling Agents - These work with AI models that can call specific functions.
2. ReAct Agents - These can work with any AI that does chat or text endpoint and deal with complex reasoning tasks.
3. Advanced Custom Agents - These use more complex methods to deal with more complex tasks and workflows.

##🚨 LlamaIndex Pitfalls from This Project (Summary provided by Augment Code)

Import & Dependency Issues
• Inconsistent module paths - Functions move between versions (draw_all_possible_flows location varies)
• Missing dependencies - Required packages not always included in base installation
• Version compatibility - Different LlamaIndex versions have breaking changes

Agent Integration Problems
• Generic responses - Agents default to general knowledge instead of using RAG documents
• Tool usage failures - Agents don't automatically use query tools without explicit prompting
• Context management - Complex workflow context handling with unclear documentation

Session Management Issues
• HTTP session leaks - HuggingFace clients don't close properly, causing memory issues
• Async cleanup problems - Background tasks create uncaught exceptions
• Resource management - No built-in cleanup mechanisms for external API clients

Provider & Model Complexity
• Payment traps - Easy to accidentally hit paid providers (provider="auto" routes to paid services)
• Provider inconsistency - Different providers have different interfaces and limitations
• Model fallback issues - No automatic fallback to free models when paid ones fail

Configuration Overhead
• Verbose setup - Requires extensive configuration for basic RAG functionality
• Manual prompt engineering - Need explicit system prompts to force proper tool usage
• Complex workflow creation - AgentWorkflow setup is not intuitive

Documentation & Examples
• Outdated examples - Many tutorials use deprecated import paths
• Missing error handling - Examples don't show proper async session management
• Unclear best practices - No clear guidance on agent vs. direct query engine usage

Performance & Reliability
• Silent failures - Agents may work but not use intended data sources
• Session buildup - Multiple queries can cause session conflicts
• Memory leaks - Improper cleanup leads to resource accumulation

Bottom Line: LlamaIndex is powerful but requires significant boilerplate code and careful configuration to work reliably in production environments.

See tutorial https://huggingface.co/learn/agents-course/en/unit2/llama-index/introduction

Notes on repo:
- Code has been updated from the tutorial to follow pythonic async/await pattern.
- Used Augment Code as the code copilot.

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
