import os
import git
import hashlib
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Other imports
# import requests
# from urllib.parse import urlparse
# import fnmatch

class GitHubRepoAnalyzer:
    """
    A comprehensive GitHub repository analyzer that uses LangChain RAG with Qdrant 
    vector database and Google Gemini LLM to track and understand code changes.
    """
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: Optional[str] = None,
        google_api_key: str = None,
        collection_name: str = "github_repo_analysis"
    ):
        """
        Initialize the GitHub Repository Analyzer.
        
        Args:
            qdrant_url: URL of the Qdrant vector database
            qdrant_api_key: API key for Qdrant (if required)
            google_api_key: Google API key for Gemini LLM
            collection_name: Name of the Qdrant collection
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.collection_name = collection_name
        
        # Initialize components
        self._initialize_clients()
        self._initialize_langchain_components()
        
        # Configuration
        self.ignored_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', 
                                 '.bin', '.obj', '.o', '.a', '.lib', '.jar', '.class',
                                 '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
                                 '.mp3', '.mp4', '.avi', '.mov', '.wav', '.pdf', '.zip'}
        
        self.ignored_patterns = ['node_modules/', '__pycache__/', '.git/', 
                               'venv/', 'env/', '.env', 'dist/', 'build/']
    
    def _initialize_clients(self):
        """Initialize Qdrant client and create collection if needed."""
        try:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            
            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                print(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            raise Exception(f"Failed to initialize Qdrant client: {str(e)}")
    
    def _initialize_langchain_components(self):
        """Initialize LangChain components."""
        if not self.google_api_key:
            raise ValueError("Google API key is required")
        
        # Initialize embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.google_api_key,
            temperature=0.3
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
    
    def _clone_repository(self, repo_url: str) -> str:
        """Clone repository to temporary directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            print(f"Cloning repository: {repo_url}")
            git.Repo.clone_from(repo_url, temp_dir ,branch='main')
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise Exception(f"Failed to clone repository: {str(e)}")
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on extension and patterns."""
        file_path_lower = file_path.lower()
        
        # Check extensions
        if any(file_path_lower.endswith(ext) for ext in self.ignored_extensions):
            return True
        
        # Check patterns
        if any(pattern in file_path for pattern in self.ignored_patterns):
            return True
        
        return False
    
    def _extract_files_content(self, repo_path: str) -> List[Dict]:
        """Extract content from all relevant files in the repository."""
        files_content = []
        repo = git.Repo(repo_path)
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      not any(pattern.rstrip('/') in d for pattern in self.ignored_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                if self._should_ignore_file(relative_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if content.strip():  # Only process non-empty files
                        # Get file stats
                        stat = os.stat(file_path)
                        
                        # Get latest commit info for this file
                        try:
                            commits = list(repo.iter_commits(paths=relative_path, max_count=1))
                            latest_commit = commits[0] if commits else None
                            commit_hash = latest_commit.hexsha if latest_commit else "unknown"
                            commit_date = latest_commit.committed_datetime if latest_commit else datetime.now()
                        except:
                            commit_hash = "unknown"
                            commit_date = datetime.now()
                        
                        files_content.append({
                            'path': relative_path,
                            'content': content,
                            'size': len(content),
                            'modified_time': datetime.fromtimestamp(stat.st_mtime),
                            'commit_hash': commit_hash,
                            'commit_date': commit_date,
                            'file_hash': hashlib.md5(content.encode()).hexdigest()
                        })
                
                except Exception as e:
                    print(f"Error reading file {relative_path}: {str(e)}")
                    continue
        
        return files_content
    
    def _create_documents(self, files_content: List[Dict], repo_url: str) -> List[Document]:
        """Create LangChain documents from file contents."""
        documents = []
        
        for file_info in files_content:
            # Split content into chunks
            chunks = self.text_splitter.split_text(file_info['content'])
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'repo_url': repo_url,
                    'file_path': file_info['path'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'file_size': file_info['size'],
                    'modified_time': file_info['modified_time'].isoformat(),
                    'commit_hash': file_info['commit_hash'],
                    'commit_date': file_info['commit_date'].isoformat(),
                    'file_hash': file_info['file_hash'],
                    'content_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
        
        return documents
    
    def analyze_repository(self, repo_url: str) -> Dict:
        """
        Analyze a GitHub repository and store in vector database.
        
        Args:
            repo_url: URL of the GitHub repository
            
        Returns:
            Dictionary with analysis results
        """
        temp_dir = None
        try:
            # Clone repository
            temp_dir = self._clone_repository(repo_url)
            
            # Extract file contents
            print("Extracting file contents...")
            files_content = self._extract_files_content(temp_dir)
            
            if not files_content:
                return {"error": "No processable files found in repository"}
            
            # Create documents
            print("Creating documents...")
            documents = self._create_documents(files_content, repo_url)
            
            # Store in vector database
            print(f"Storing {len(documents)} documents in vector database...")
            self.vector_store.add_documents(documents)
            
            # Generate summary using LLM
            summary = self._generate_repository_summary(files_content, repo_url)
            
            result = {
                "repo_url": repo_url,
                "analysis_date": datetime.now().isoformat(),
                "total_files": len(files_content),
                "total_chunks": len(documents),
                "files_analyzed": [f['path'] for f in files_content],
                "summary": summary,
                "status": "success"
            }
            
            print("Repository analysis completed successfully!")
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "status": "failed"}
        
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _generate_repository_summary(self, files_content: List[Dict], repo_url: str) -> str:
        """Generate a summary of the repository using LLM."""
        try:
            # Prepare context
            file_types = {}
            total_lines = 0
            
            for file_info in files_content:
                ext = Path(file_info['path']).suffix or 'no_extension'
                file_types[ext] = file_types.get(ext, 0) + 1
                total_lines += file_info['content'].count('\n')
            
            context = f"""
            Repository: {repo_url}
            Total files: {len(files_content)}
            Total lines of code: {total_lines}
            File types: {dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True))}
            
            Sample files:
            {chr(10).join([f"- {f['path']}" for f in files_content[:10]])}
            """
            
            prompt = f"""
            Analyze this code repository and provide a comprehensive summary:
            
            {context}
            
            Please provide:
            1. Main purpose and functionality of the repository
            2. Key technologies and frameworks used
            3. Architecture overview
            4. Notable patterns or interesting aspects
            5. Potential areas for improvement
            
            Keep the summary concise but informative.
            """
            
            summary = self.llm(prompt)
            return summary
            
        except Exception as e:
            return f"Could not generate summary: {str(e)}"
    
    def detect_changes(self, repo_url: str) -> Dict:
        """
        Detect changes in repository compared to stored version.
        
        Args:
            repo_url: URL of the GitHub repository
            
        Returns:
            Dictionary with change detection results
        """
        temp_dir = None
        try:
            # Clone current version
            temp_dir = self._clone_repository(repo_url)
            current_files = self._extract_files_content(temp_dir)
            
            # Query existing documents from vector store
            existing_docs = self._get_existing_documents(repo_url)
            
            if not existing_docs:
                return {"error": "No existing analysis found for this repository"}
            
            # Compare versions
            changes = self._compare_versions(existing_docs, current_files)
            
            # Update vector store with changes
            if changes['modified_files'] or changes['new_files']:
                self._update_vector_store(repo_url, current_files, changes)
            
            # Generate change analysis using LLM
            change_analysis = self._analyze_changes_with_llm(changes)
            
            result = {
                "repo_url": repo_url,
                "analysis_date": datetime.now().isoformat(),
                "changes": changes,
                "change_analysis": change_analysis,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Change detection failed: {str(e)}", "status": "failed"}
        
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _get_existing_documents(self, repo_url: str) -> List[Dict]:
        """Get existing documents for a repository from vector store."""
        try:
            # Search for documents with matching repo_url
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.repo_url",
                            match=models.MatchValue(value=repo_url)
                        )
                    ]
                ),
                limit=10000  # Adjust based on expected repository size
            )
            
            return [point.payload for point in search_result[0]]
            
        except Exception as e:
            print(f"Error retrieving existing documents: {str(e)}")
            return []
    
    def _compare_versions(self, existing_docs: List[Dict], current_files: List[Dict]) -> Dict:
        """Compare existing and current versions to detect changes."""
        # Create lookup dictionaries
        existing_files = {}
        for doc in existing_docs:
            file_path = doc.get('file_path')
            if file_path and doc.get('chunk_index') == 0:  # Use first chunk metadata
                existing_files[file_path] = {
                    'file_hash': doc.get('file_hash'),
                    'commit_hash': doc.get('commit_hash'),
                    'modified_time': doc.get('modified_time')
                }
        
        current_files_dict = {f['path']: f for f in current_files}
        
        # Detect changes
        new_files = []
        modified_files = []
        deleted_files = []
        
        # Check for new and modified files
        for file_path, file_info in current_files_dict.items():
            if file_path not in existing_files:
                new_files.append(file_path)
            else:
                existing_hash = existing_files[file_path]['file_hash']
                if file_info['file_hash'] != existing_hash:
                    modified_files.append({
                        'path': file_path,
                        'old_hash': existing_hash,
                        'new_hash': file_info['file_hash'],
                        'old_commit': existing_files[file_path]['commit_hash'],
                        'new_commit': file_info['commit_hash']
                    })
        
        # Check for deleted files
        for file_path in existing_files:
            if file_path not in current_files_dict:
                deleted_files.append(file_path)
        
        return {
            'new_files': new_files,
            'modified_files': modified_files,
            'deleted_files': deleted_files,
            'total_changes': len(new_files) + len(modified_files) + len(deleted_files)
        }
    
    def _update_vector_store(self, repo_url: str, current_files: List[Dict], changes: Dict):
        """Update vector store with changed files."""
        try:
            # Remove old documents for modified and deleted files
            files_to_remove = [f['path'] for f in changes['modified_files']] + changes['deleted_files']
            
            if files_to_remove:
                for file_path in files_to_remove:
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="metadata.repo_url",
                                        match=models.MatchValue(value=repo_url)
                                    ),
                                    models.FieldCondition(
                                        key="metadata.file_path",
                                        match=models.MatchValue(value=file_path)
                                    )
                                ]
                            )
                        )
                    )
            
            # Add new and modified files
            files_to_add = []
            current_files_dict = {f['path']: f for f in current_files}
            
            for file_path in changes['new_files']:
                files_to_add.append(current_files_dict[file_path])
            
            for modified_file in changes['modified_files']:
                files_to_add.append(current_files_dict[modified_file['path']])
            
            if files_to_add:
                documents = self._create_documents(files_to_add, repo_url)
                self.vector_store.add_documents(documents)
            
            print(f"Updated vector store: {len(files_to_add)} files processed")
            
        except Exception as e:
            print(f"Error updating vector store: {str(e)}")
    
    def _analyze_changes_with_llm(self, changes: Dict) -> str:
        """Analyze changes using LLM."""
        try:
            if changes['total_changes'] == 0:
                return "No changes detected in the repository."
            
            prompt = f"""
            Analyze the following code repository changes:
            
            New files: {len(changes['new_files'])}
            {chr(10).join([f"- {f}" for f in changes['new_files'][:5]])}
            
            Modified files: {len(changes['modified_files'])}
            {chr(10).join([f"- {f['path']}" for f in changes['modified_files'][:5]])}
            
            Deleted files: {len(changes['deleted_files'])}
            {chr(10).join([f"- {f}" for f in changes['deleted_files'][:5]])}
            
            Provide insights about:
            1. The nature and scope of changes
            2. Potential impact on the codebase
            3. Whether changes suggest new features, bug fixes, or refactoring
            4. Any patterns or trends in the modifications
            
            Keep the analysis concise but insightful.
            """
            
            analysis = self.llm(prompt)
            return analysis
            
        except Exception as e:
            return f"Could not analyze changes: {str(e)}"
    
    def query_repository(self, query: str, repo_url: Optional[str] = None) -> str:
        """
        Query the repository knowledge base.
        
        Args:
            query: Question about the repository
            repo_url: Optional specific repository URL to query
            
        Returns:
            Answer based on repository analysis
        """
        try:
            # Create custom prompt template
            template = """
            You are an expert code analyst. Use the following context from a code repository to answer the question.
            
            Context: {context}
            
            Question: {question}
            
            Provide a detailed and accurate answer based on the code context. If the information isn't available in the context, say so clearly.
            
            Answer:
            """
            
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            
            # Create retrieval chain
            if repo_url:
                # Filter by specific repository
                retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "filter": {"repo_url": repo_url},
                        "k": 5
                    }
                )
            else:
                retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            answer = qa_chain.run(query)
            return answer
            
        except Exception as e:
            return f"Query failed: {str(e)}"
    
    def get_repository_stats(self, repo_url: str) -> Dict:
        """Get statistics about analyzed repository."""
        try:
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.repo_url",
                            match=models.MatchValue(value=repo_url)
                        )
                    ]
                ),
                limit=10000
            )
            
            points = search_result[0]
            if not points:
                return {"error": "Repository not found in database"}
            
            # Aggregate statistics
            files = set()
            total_chunks = len(points)
            file_types = {}
            latest_analysis = None
            
            for point in points:
                metadata = point.payload
                files.add(metadata.get('file_path'))
                
                file_path = metadata.get('file_path', '')
                ext = Path(file_path).suffix or 'no_extension'
                file_types[ext] = file_types.get(ext, 0) + 1
                
                # Track latest analysis
                modified_time = metadata.get('modified_time')
                if modified_time and (not latest_analysis or modified_time > latest_analysis):
                    latest_analysis = modified_time
            
            return {
                "repo_url": repo_url,
                "total_files": len(files),
                "total_chunks": total_chunks,
                "file_types": dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True)),
                "latest_analysis": latest_analysis,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Failed to get repository stats: {str(e)}"}


# Example usage
if __name__ == "__main__":
    # Configuration

    
    # Initialize analyzer
    analyzer = GitHubRepoAnalyzer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        collection_name=COLLECTION_NAME
    )
    
    # Example repository URL
    repo_url = "https://github.com/openai/openai-cs-agents-demo"
    repo_url = "https://github.com/jayanth119/refactored-JobSheet"
    
    # Analyze repository
    print("Analyzing repository...")
    result = analyzer.analyze_repository(repo_url)
    print(f"Analysis result: {result}")
    
    # Detect changes (run this later to detect changes)
    print("\nDetecting changes...")
    changes = analyzer.detect_changes(repo_url)
    print(f"Changes detected: {changes}")
    
    # Query repository
    # print("\nQuerying repository...")
    # answer = analyzer.query_repository("What is the main functionality of this repository?", repo_url)
    # print(f"Answer: {answer}")
    
    # Get repository statistics
    print("\nRepository statistics...")
    stats = analyzer.get_repository_stats(repo_url)
    print(f"Stats: {stats}")