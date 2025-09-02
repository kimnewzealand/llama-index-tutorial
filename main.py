"""
LlamaIndex Tutorial: RAG Application with ChromaDB and Agent Workflow
A Pythonic implementation of a Retrieval-Augmented Generation system with an Agent Workflow.

🆓 FREE & OPEN SOURCE ONLY - No paid services required!

This application uses only free models and services:
- HuggingFace free models (DialoGPT, GPT-2, etc.)
- Free HuggingFace API token (no payment required)
- Local ChromaDB vector storage
- Local document processing

For completely local setup (no internet required):
1. Replace HuggingFaceInferenceAPI with local models using Ollama
2. Use local embedding models with sentence-transformers
3. All data stays on your machine
"""

import asyncio
import logging
import os
import sys
import warnings
import atexit
from pathlib import Path
from typing import Optional

# Suppress specific warnings related to async session cleanup
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Task was destroyed but it is pending.*")
warnings.filterwarnings("ignore", message=".*Deleting.*client but some sessions are still open.*")
warnings.filterwarnings("ignore", message=".*Unclosed client session.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="aiohttp")

# Suppress all HuggingFace and aiohttp session warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

# Additional suppression for specific modules
import sys
import io
from contextlib import redirect_stderr

# Monkey patch to suppress aiohttp session warnings
original_stderr = sys.stderr

class SuppressedStderr:
    def write(self, s):
        if "Task exception was never retrieved" not in s and "KeyError" not in s:
            original_stderr.write(s)
    def flush(self):
        original_stderr.flush()

# Apply the suppression
sys.stderr = SuppressedStderr()

# Also suppress asyncio and aiohttp warnings
import logging
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants - ALL FREE MODELS
DEFAULT_INFERENCE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Free conversational model
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"    # Free embedding model

# Free fallback model alternatives
FREE_LLM_MODELS = [
    "microsoft/DialoGPT-medium",    # Best free conversational model
    "microsoft/DialoGPT-small",     # Smaller, faster free model
    "gpt2",                         # Classic free model, very reliable
    "distilgpt2",                   # Lightweight version of GPT-2
    "facebook/blenderbot-400M-distill"  # Free chatbot model
]

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
CHROMA_DB_PATH = "./alfred_chroma_db"
COLLECTION_NAME = "alfred"


# Global registry for tracking clients that need cleanup
_active_clients = []

def register_client_for_cleanup(client):
    """Register a client for cleanup on exit."""
    _active_clients.append(client)

def cleanup_all_clients():
    """Clean up all registered clients."""
    # Suppress all output during cleanup
    with redirect_stderr(io.StringIO()):
        for client in _active_clients:
            try:
                if hasattr(client, 'close'):
                    if asyncio.iscoroutinefunction(client.close):
                        # Can't await in atexit, so just try to close synchronously
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(client.close())
                        except Exception:
                            pass
                    else:
                        client.close()
            except Exception:
                pass  # Ignore cleanup errors
        _active_clients.clear()

        # Force garbage collection
        import gc
        gc.collect()

# Register cleanup function to run on exit
atexit.register(cleanup_all_clients)

def setup_environment() -> None:
    """Setup environment variables and configurations."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    load_dotenv()


def get_hf_token() -> str:
    """Get HuggingFace token from environment variables.

    Returns:
        HuggingFace token

    Raises:
        SystemExit: If token is not found
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not found in environment variables")
        logger.error("Please create a .env file with your HuggingFace token:")
        logger.error("HF_TOKEN=your_huggingface_api_key_here")
        logger.error("Get your FREE token from: https://huggingface.co/settings/tokens")
        logger.error("")
        logger.error("💡 This application uses ONLY FREE models - no payment required!")
        logger.error("💡 The token is free and just needed for API access limits")
        sys.exit(1)

    return hf_token


class RAGApplication:
    """A Retrieval-Augmented Generation application using LlamaIndex and ChromaDB."""

    def __init__(self, hf_token: str):
        """Initialize the RAG application.

        Args:
            hf_token: HuggingFace API token
        """
        self.hf_token = hf_token
        self.llm: Optional[HuggingFaceInferenceAPI] = None
        self.embed_model: Optional[HuggingFaceEmbedding] = None
        self.vector_store: Optional[ChromaVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self.pipeline: Optional[IngestionPipeline] = None


    def setup_llm(self, model_name: str = DEFAULT_INFERENCE_MODEL) -> bool:
        """Setup the Language Model with free model fallbacks.

        Args:
            model_name: Name of the HuggingFace model to use

        Returns:
            True if successful, False otherwise
        """
        # List of free models to try in order
        free_models = [model_name] + FREE_LLM_MODELS
        # Remove duplicates while preserving order
        free_models = list(dict.fromkeys(free_models))

        for model in free_models:
            try:
                logger.info(f"Trying free model: {model}")

                # Close any existing LLM client before creating a new one
                if self.llm and hasattr(self.llm, '_client'):
                    try:
                        if hasattr(self.llm._client, 'close'):
                            if asyncio.iscoroutinefunction(self.llm._client.close):
                                # We can't await here, so we'll schedule it
                                loop = asyncio.get_event_loop()
                                loop.create_task(self.llm._client.close())
                    except Exception:
                        pass  # Ignore cleanup errors during initialization

                self.llm = HuggingFaceInferenceAPI(
                    model_name=model,
                    temperature=0.7,
                    max_tokens=100,
                    token=self.hf_token,
                    provider="hyperbolic"
                )

                # Register the client for cleanup
                if hasattr(self.llm, '_client'):
                    register_client_for_cleanup(self.llm._client)
                if hasattr(self.llm, '_async_client'):
                    register_client_for_cleanup(self.llm._async_client)

                logger.info(f"✅ Successfully initialized free LLM: {model}")
                return True
            except Exception as e:
                logger.warning(f"❌ Failed to initialize {model}: {e}")
                continue

        logger.error("❌ Failed to initialize any free LLM model")
        return False

    def setup_vector_store(self, db_path: str = CHROMA_DB_PATH) -> bool:
        """Setup ChromaDB vector store.

        Args:
            db_path: Path to ChromaDB database

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create database directory if it doesn't exist
            Path(db_path).mkdir(parents=True, exist_ok=True)

            db = chromadb.PersistentClient(path=db_path)
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            logger.info(f"Successfully initialized vector store at: {db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False

    def setup_embedding_model(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> bool:
        """Setup embedding model.

        Args:
            model_name: Name of the embedding model

        Returns:
            True if successful, False otherwise
        """
        try:
            self.embed_model = HuggingFaceEmbedding(model_name=model_name)
            logger.info(f"Successfully initialized embedding model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False

    def setup_pipeline(self,
                      chunk_size: int = DEFAULT_CHUNK_SIZE,
                      chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> bool:
        """Setup ingestion pipeline.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            True if successful, False otherwise
        """
        if not all([self.embed_model, self.vector_store]):
            logger.error("Embedding model and vector store must be initialized first")
            return False

        try:
            self.pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                    self.embed_model,
                ],
                vector_store=self.vector_store,
            )

            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )

            logger.info("Successfully initialized ingestion pipeline and index")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize all components of the RAG application.

        Returns:
            True if all components initialized successfully, False otherwise
        """
        logger.info("Initializing RAG application...")

        steps = [
            ("LLM", self.setup_llm),
            ("Vector Store", self.setup_vector_store),
            ("Embedding Model", self.setup_embedding_model),
            ("Pipeline", self.setup_pipeline),
        ]

        for step_name, setup_func in steps:
            if not setup_func():
                logger.error(f"Failed to initialize {step_name}")
                return False

        logger.info("RAG application initialized successfully!")
        return True

    def test_llm(self, prompt: str = "Hello, how are you?") -> None:
        """Test the LLM with a simple prompt.

        Args:
            prompt: Test prompt to send to the LLM
        """
        if not self.llm:
            logger.warning("LLM not initialized, attempting to set it up...")
            if not self.setup_llm():
                logger.error("Failed to setup LLM for testing")
                return

        try:
            logger.info(f"Testing LLM with prompt: {prompt}")
            response = self.llm.complete(prompt)
            logger.info(f"✅ Test passed LLM Response: {response}")
        except Exception as e:
            logger.error(f"❌ LLM test failed: {e}")
            logger.info("Attempting to reinitialize LLM...")
            if self.setup_llm():
                try:
                    response = self.llm.complete(prompt)
                    logger.info(f"LLM Response after reinit: {response}")
                except Exception as retry_e:
                    logger.error(f"❌LLM test failed even after reinit: {retry_e}")
            else:
                logger.error("Failed to reinitialize LLM")

    async def cleanup(self) -> None:
        """Clean up resources and close any open sessions."""
        try:
            if self.llm and hasattr(self.llm, '_client'):
                client = self.llm._client
                if client:
                    logger.debug("Closing HuggingFace client sessions...")
                    try:
                        # Suppress stderr during cleanup to avoid KeyError messages
                        with redirect_stderr(io.StringIO()):
                            if hasattr(client, 'close'):
                                await client.close()
                                logger.debug("✅ Client closed with close() method")
                            elif hasattr(client, 'aclose'):
                                await client.aclose()
                                logger.debug("✅ Client closed with aclose() method")
                    except Exception as close_error:
                        # Silently ignore cleanup errors
                        pass

            # Also try to access and close any internal async client
            if self.llm and hasattr(self.llm, '_async_client'):
                async_client = self.llm._async_client
                if async_client:
                    try:
                        with redirect_stderr(io.StringIO()):
                            if hasattr(async_client, 'close'):
                                await async_client.close()
                                logger.debug("✅ Async client closed")
                    except Exception:
                        pass

            # Force garbage collection to clean up any remaining sessions
            import gc
            gc.collect()

            # Small delay to allow cleanup to complete
            await asyncio.sleep(0.1)
            logger.debug("✅ Cleanup completed")
        except Exception as e:
            # Silently ignore all cleanup errors
            pass

    def load_data(self, data_path: str = "./documents") -> None:
        """Load data into the vector store.

        Args:
            data_path: Path to the directory containing documents
        """
        try:
            logger.info(f"RAG Stage 1: Loading data from {data_path}")
            reader = SimpleDirectoryReader(input_dir=str(data_path))
            documents = reader.load_data()

            if not documents:
                logger.warning("No documents found to load")
                return

            logger.info(f"Loaded {len(documents)} documents")

            # Process documents through the pipeline if available
            if self.pipeline:
                logger.info("RAG Stage 2: Indexing documents")
                self.nodes = self.pipeline.run(documents=documents)
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    embed_model=self.embed_model
                )
                self.index.insert_nodes(self.nodes)
                logger.info(f"RAG Stage 3: Stored {len(self.nodes)} nodes")
            else:
                logger.warning("Pipeline not available, documents loaded but not processed")

        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            logger.info("💡 Tip: Make sure the documents directory exists and contains readable files")
        

async def main() -> None:
    """Main function to run the RAG application."""
    try:
        # Setup environment
        setup_environment()

        # Get HuggingFace token
        hf_token = get_hf_token()

        # Initialize RAG application
        rag_app = RAGApplication(hf_token)

        if rag_app.initialize():
            # Test the LLM
            rag_app.test_llm()
            logger.info("RAG application is ready for use!")

            # Load data
            rag_app.load_data()

            # Query the index
            logger.info("RAG Stage 4 - Querying the index")
            query_engine = rag_app.index.as_query_engine(
                    llm=rag_app.llm,
                    response_mode="tree_summarize",
            )
            response = query_engine.query("🔍How many levels of data classification are there?")
            print(response)
            logger.info("RAG Stage 5 - Evaluating the response")
            evaluator = FaithfulnessEvaluator(llm=rag_app.llm)
            eval_result = evaluator.evaluate_response(response=response)
            print(f"Faithfulness Evaluation result: {eval_result.passing}")
        else:
            logger.error("Failed to initialize RAG application")
            sys.exit(1)
        try:
            # Use AgentWorkFlow, an orchestrator for running a system of one or more agents.
            logger.info("Initializing agent...")
            query_tool = QueryEngineTool.from_defaults(query_engine, name="Query engine tool", description="tool to use to answer questions on the compliance policies")
            agent = AgentWorkflow.from_tools_or_functions(
                [query_tool],
                llm=rag_app.llm,
            )
            logger.info("Agent initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
        try:
            #  Agents are stateless by default however, they can remember past interactions using a Context object
            logger.info(" Running agent...")

            # Create context for the agent
            ctx = Context(agent)

            # First query with proper async handling
            logger.info("🔍 Asking: How many levels of data classification are there?")
            try:
                result = await agent.run("How many levels of data classification are there?", ctx=ctx)
                if hasattr(result, 'response'):
                    print(f"🎯 Agent response: {result.response}")
                elif hasattr(result, 'content'):
                    print(f"🎯 Agent response: {result.content}")
                else:
                    print(f"🎯 Agent result: {result}")

                # Immediate cleanup after first query to prevent session buildup
                await rag_app.cleanup()
                await asyncio.sleep(0.5)  # Allow cleanup to complete

            except Exception as query_error:
                logger.error(f"First query failed: {query_error}")
                # Try without context as fallback
                try:
                    result = await agent.run("How many levels of data classification are there?")
                    print(f"🎯 Agent response (no context): {result}")
                except Exception as fallback_error:
                    logger.error(f"Fallback query also failed: {fallback_error}")

            # Second query with fresh session
            logger.info("🔍 Asking: Why are there 3 levels of data classification?")
            try:
                # Create fresh context to avoid session conflicts
                ctx2 = Context(agent)
                result = await agent.run("Why are there 3 levels of data classification?", ctx=ctx2)
                if hasattr(result, 'response'):
                    print(f"🎯 Agent response: {result.response}")
                elif hasattr(result, 'content'):
                    print(f"🎯 Agent response: {result.content}")
                else:
                    print(f"🎯 Agent result: {result}")
            except Exception as query_error:
                logger.error(f"Second query failed: {query_error}")
                logger.info("💡 This might be due to session conflicts or payment provider issues")

                # Try direct query engine as fallback
                try:
                    logger.info("🔧 Trying direct query engine...")
                    direct_response = query_engine.query("Why are there 3 levels of data classification?")
                    print(f"🔧 Direct query engine response: {direct_response}")
                except Exception as direct_error:
                    logger.error(f"Direct query engine also failed: {direct_error}")

            # Final cleanup after agent operations
            await rag_app.cleanup()

        except Exception as e:
            logger.error(f"Failed to run agent: {e}")
            logger.info("💡 Tip: Trying direct query engine as fallback...")

            # Try direct query engine as fallback
            try:
                logger.info("Testing query engine directly...")
                direct_response = query_engine.query("How many levels of data classification are there?")
                print(f"🔧 Direct query engine response: {direct_response}")
            except Exception as direct_error:
                logger.error(f"Direct query engine also failed: {direct_error}")

            # Cleanup even if there's an error
            try:
                await rag_app.cleanup()
            except Exception:
                pass


    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean shutdown to prevent session cleanup errors
        logger.info("Performing cleanup...")
        try:
            if 'rag_app' in locals():
                await rag_app.cleanup()
        except Exception as cleanup_error:
            logger.debug(f"Cleanup error (non-critical): {cleanup_error}")

        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main())

