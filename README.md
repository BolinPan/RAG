# Retrieval-Augmented Generation (RAG) Learning Repository

This repository provides a comprehensive, hands-on exploration of Retrieval-Augmented Generation (RAG) techniques and strategies. Retrieval-Augmented Generation has emerged as a powerful mechanism to expand an LLM's knowledge base by retrieving relevant documents from external data sources and grounding LLM generation via in-context learning. These notebooks progressively build understanding from foundational RAG concepts to advanced orchestration strategies.

## Overview

RAG combines the strengths of both retrieval and generation:
- **Retrieval**: Fetches relevant documents from a knowledge base using semantic similarity or other ranking methods.
- **Augmentation**: Incorporates retrieved documents into the LLM prompt to provide factual grounding.
- **Generation**: The LLM generates answers informed by the retrieved context, reducing hallucinations and improving factuality.

This repository is organized into six progressive sections, each building on core concepts to reach sophisticated multi-agent systems.

## Repository Structure

### Section 1: Naive RAG
**Files**: `Section_1_NaiveRAG/NaiveRAG.ipynb`

The foundational RAG pipeline. This section covers:
- Loading documents from web sources
- Splitting documents into chunks
- Creating embeddings with OpenAI
- Building a vector database (Chroma)
- Simple retrieval and answer generation using a basic RAG chain

**Key Concepts**: Document loading, chunking, embeddings, vector search, RAG prompts.

---

### Section 2: Query Translation & Transformation
**Files**: `Section_2_QueryTranslation/`
1. **MultiQueryRetriever.ipynb** - Generates multiple query perspectives to improve retrieval coverage
2. **RRFMultiQueryRetriever.ipynb** - Combines multi-query results using Reciprocal Rank Fusion for better ranking
3. **DecomposedQueryRetriever.ipynb** - Breaks complex questions into sub-problems for targeted retrieval
4. **StepBackQueryTransformation.ipynb** - Paraphrases queries into more generic foundational questions
5. **HyDE.ipynb** - Uses LLM to generate hypothetical documents that improve retrieval alignment

**Key Concepts**: Query optimization, multi-perspective retrieval, query decomposition, hypothetical document embeddings.

These techniques improve retrieval quality by transforming or multiplying queries rather than relying on a single retrieval attempt.

---

### Section 3: Routing & Query Construction
**Files**: `Section_3_Routing_QueryConstruction/`
1. **LogicalRouting.ipynb** - Routes questions to different datasources based on structured LLM classification
2. **SemanticRouting.ipynb** - Uses embedding similarity to route queries to specialized experts/prompts
3. **QueryStructuring.ipynb** - Converts free-text questions into structured database queries using LLM

**Key Concepts**: Query routing, datasource selection, semantic similarity, structured outputs, query schemas.

These techniques enable the RAG system to intelligently direct queries to the most appropriate processing pipeline.

---

### Section 4: Indexing & Storage
**Files**: `Section_4_Indexing/`
1. **MultiRepresentationIndexing.ipynb** - Stores multiple representations (summaries) of documents to improve retrieval quality while maintaining access to full context

**Key Concepts**: Document summarization, multi-vector storage, representation diversity, retriever linking.

Advanced indexing strategies that optimize both retrieval quality and context preservation.

---

### Section 5: Retrieval Enhancement
**Files**: `Section_5_Retrieval/`
1. **Reranking.ipynb** - Uses specialized reranking models (Cohere) to refine initial retrieval results and improve ranking quality

**Key Concepts**: Post-retrieval ranking, relevance scoring, contextual compression, ranking models.

Post-processing techniques that refine retrieval results after initial similarity-based ranking.

---

### Section 6: Advanced RAG Strategies
**Files**: `Section_6_AdvancedRAG/`

Advanced orchestration and decision-making patterns using LangGraph:

1. **CRAG.ipynb** - Conditional Retrieval and Generation
   - Grades document relevance and hallucinatory answers
   - Conditionally rewrites queries when retrieval is insufficient
   - Integrates web search as a fallback mechanism
   - Uses structured LLM outputs for quality assessment

2. **AdaptiveRAG.ipynb** - Adaptive RAG with Multi-Step Orchestration
   - Routes queries to vectorstore or web search
   - Implements five independent grading classifiers
   - Evaluates hallucinations and answer relevance
   - Adapts behavior based on document and generation quality
   - Compiles into a state-based graph execution engine

3. **SelfRAG.ipynb** - Self-Reflective RAG
   - LLM grades its own retrieved documents for relevance
   - Validates generated answers for hallucinations
   - Checks if answers address the user question
   - Iteratively refines retrieval with query rewriting
   - Self-evaluation loop for quality improvement

4. **AgenticRAG.ipynb** - Agentic RAG with Tool-Calling
   - LLM agent autonomously decides when to retrieve
   - Uses tools (retriever) through natural LLM reasoning
   - Evaluates document relevance after retrieval
   - Rewrites queries for failed retrievals
   - Message-based history for multi-turn conversation

**Key Concepts**: Graph-based orchestration, structured grading, hallucination detection, query rewriting, web search integration, agentic decision-making, tool calling, adaptive routing.

---

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key
- Tavily API key (for web search in advanced examples)
- Cohere API key (for reranking in advanced examples)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd RAG

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp configs/config.yaml.example configs/config.yaml
# Edit configs/config.yaml with your API keys
```

### Configuration
Create a `configs/config.yaml` file with your API credentials:

```yaml
API:
  OPENAI: your-openai-api-key
  LANGCHAIN: your-langchain-api-key
  TAVILY: your-tavily-api-key (optional, for web search)
  COHERE: your-cohere-api-key (optional, for reranking)
```

### Running Notebooks

Start with **Section 1** to understand basic RAG, then progress through sections:

```bash
# For Naive RAG
jupyter notebook Section_1_NaiveRAG/NaiveRAG.ipynb

# For Query Translation techniques
jupyter notebook Section_2_QueryTranslation/1.\ MultiQueryRetriever.ipynb

# For advanced strategies
jupyter notebook Section_6_AdvancedRAG/2.\ AdaptiveRAG.ipynb
```

---

## Learning Path

### For Beginners
1. **Section 1**: Understand basic RAG mechanics
2. **Section 2, Part 1**: Learn simple query improvements (MultiQueryRetriever)
3. **Section 3, Part 1**: Understand routing concepts

### For Intermediate Users
1. All of **Section 2**: Query transformation techniques
2. All of **Section 3**: Routing and query construction
3. **Section 5**: Retrieval enhancement with reranking

### For Advanced Users
1. **Section 4**: Multi-representation indexing strategies
2. All of **Section 6**: Advanced orchestration with LangGraph
   - Start with CRAG for conditional logic
   - Progress to AdaptiveRAG for multi-step evaluation
   - Explore SelfRAG for self-reflection
   - Understand AgenticRAG for autonomous decision-making

---

## Key Technologies & Tools

| Technology | Purpose | Used In |
|-----------|---------|---------|
| **LangChain** | LLM orchestration & prompts | All sections |
| **OpenAI GPT** | LLM generation & classification | All sections |
| **Chroma** | Vector database | Sections 1-6 |
| **LangGraph** | State-based workflow orchestration | Section 6 |
| **Tavily** | Web search integration | Section 6 (CRAG, AdaptiveRAG) |
| **Cohere** | Document reranking | Section 5 & 6 |

---

## Core RAG Concepts Covered

### Document Retrieval
- Vector similarity search
- Multi-vector retrieval
- Semantic routing
- Query transformation
- Hypothetical document embeddings

### Query Processing
- Query decomposition
- Step-back prompting
- Query rewriting
- Logical routing
- Semantic routing
- Query structuring

### Quality Assurance
- Document relevance grading
- Hallucination detection
- Answer relevance evaluation
- Self-reflection mechanisms
- Iterative refinement

### Advanced Orchestration
- Conditional workflows
- State management
- Tool-based agents
- Multi-step decision logic
- Adaptive branching
- Web search fallback

---

## References & Inspiration

This repository implements techniques from:
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Advanced RAG Papers](https://arxiv.org/abs/2310.03025)
- [Self-RAG: Learning to Retrieve, Generate, and Critique for Self-Improving RAG](https://arxiv.org/abs/2310.11609)
- [CRAG: Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)

---

## Notes

- Each notebook is self-contained and can be run independently
- Configuration credentials are loaded from `configs/config.yaml`
- Some notebooks require API keys for external services (OpenAI, Tavily, Cohere)
- Advanced notebooks (Section 6) demonstrate production-ready patterns with full error handling and conditional logic

---

## Contributing

Contributions, suggestions, and improvements are welcome! Please feel free to:
- Open issues for bugs or questions
- Submit pull requests for enhancements
- Suggest new RAG techniques to explore

---

## License

This repository is provided as-is for educational purposes.
