import logging
import os
import sys
from pathlib import Path
from typing import Optional

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_INFERENCE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
CHROMA_DB_PATH = "./alfred_chroma_db"
COLLECTION_NAME = "alfred"

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
        logger.error("Get your token from: https://huggingface.co/settings/tokens")
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
        """Setup the Language Model.

        Args:
            model_name: Name of the HuggingFace model to use

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Initializing LLM: {model_name}")
            self.llm = HuggingFaceInferenceAPI(
                model_name=model_name,
                temperature=0.7,
                max_tokens=100,
                token=self.hf_token,
                provider="auto"
            )
            logger.info(f"Successfully initialized LLM: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM {model_name}: {e}")
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
        

def main() -> None:
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
            response = query_engine.query("How many levels of data classification are there?")
            print(response)
            logger.info("RAG Stage 5 - Evaluating the response")
            evaluator = FaithfulnessEvaluator(llm=rag_app.llm)
            eval_result = evaluator.evaluate_response(response=response)
            print(f"Faithfulness Evaluation result: {eval_result.passing}")
        else:
            logger.error("Failed to initialize RAG application")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

