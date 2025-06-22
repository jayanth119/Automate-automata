import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

# Core dependencies
from github import Github
from git import Repo
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPRAnalyzer:
    def __init__(self):
        """Initialize the PR analyzer with all necessary configurations"""
        load_dotenv()

        # Environment variables
        add 

        if not self.github_pat or not self.gemini_api_key:
            raise ValueError("Missing required environment variables: GITHUB_PAT, GEMINI_API_KEY")

        # GitHub setup
        self.github = Github(self.github_pat)
        self.repo = self.github.get_repo(self.repo_name)

        # Gemini setup with LangChain
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.gemini_api_key,
            temperature=0.3,
            max_output_tokens=2048
        )

        # Configuration
        self.state_file = "pr_state.json"
        self.temp_dir = Path("/tmp/pr_analysis")
        self.temp_dir.mkdir(exist_ok=True)

        # Text splitter for large codebases
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # File extensions to analyze
        self.code_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go',
            '.rs', '.php', '.rb', '.scala', '.kt', '.swift', '.r', '.m', '.mm',
            '.html', '.css', '.scss', '.less', '.vue', '.jsx', '.tsx',
            '.json', '.yaml', '.yml', '.xml', '.toml',
            '.md', '.rst', '.txt'
        }

    def load_previous_prs(self) -> List[int]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading previous PR state: {e}")
        return []

    def save_current_prs(self, pr_numbers: List[int]) -> None:
        try:
            with open(self.state_file, "w") as f:
                json.dump(pr_numbers, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving PR state: {e}")

    def get_new_prs(self) -> List[Any]:
        previous_prs = self.load_previous_prs()
        open_prs = list(self.repo.get_pulls(state='open'))
        current_pr_numbers = [pr.number for pr in open_prs]
        new_prs = [pr for pr in open_prs if pr.number not in previous_prs]
        self.save_current_prs(current_pr_numbers)
        return new_prs

    def clone_pr_repo(self, pr: Any) -> Path:
        commit_sha = pr.head.sha
        clone_url = pr.head.repo.clone_url
        local_path = self.temp_dir / f"pr_{pr.number}"
        try:
            if local_path.exists():
                shutil.rmtree(local_path)
            logger.info(f"Cloning from {clone_url}")
            repo_local = Repo.clone_from(clone_url, local_path)
            repo_local.git.checkout(commit_sha)
            logger.info(f"Checked out to commit: {commit_sha}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to clone repo for PR #{pr.number}: {e}")
            raise

    def extract_code_files(self, repo_path: Path) -> List[Dict[str, str]]:
        files_data = []
        for file_path in repo_path.rglob("*"):
            if (file_path.is_file() and
                file_path.suffix.lower() in self.code_extensions and
                not any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules', '.venv', 'venv'])):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    relative_path = file_path.relative_to(repo_path)
                    files_data.append({
                        'path': str(relative_path),
                        'content': content,
                        'extension': file_path.suffix,
                        'size': len(content)
                    })
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
        return files_data

    def create_analysis_prompts(self) -> Dict[str, PromptTemplate]:
        return {
            'file_analysis': PromptTemplate(
                input_variables=["file_path", "file_content"],
                template="""
Analyze the following code file and provide insights:

File: {file_path}

Code:
{file_content}

Please provide:
1. Purpose
2. Key Components
3. Code Quality
4. Security Concerns
5. Performance Issues
6. Suggestions
                """
            ),
            'repo_summary': PromptTemplate(
                input_variables=["file_analyses"],
                template="""
Given the following file-level analyses, summarize the repository:

{file_analyses}

Please provide:
1. Repository Overview
2. Technology Stack
3. Code Quality Assessment
4. Security Analysis
5. Performance Review
6. Architecture Insights
7. Recommendations
8. Complexity Assessment
                """
            ),
            'pr_analysis': PromptTemplate(
                input_variables=["pr_title", "pr_description", "pr_author", "repo_analysis"],
                template="""
Analyze this Pull Request:

Title: {pr_title}
Author: {pr_author}
Description: {pr_description}

Repository Context:
{repo_analysis}

Please provide:
1. PR Impact Assessment
2. Change Quality
3. Risk Analysis
4. Integration Concerns
5. Review Recommendations
6. Testing Suggestions
7. Documentation Needs
                """
            )
        }

    def analyze_files(self, files_data: List[Dict[str, str]]) -> List[str]:
        prompts = self.create_analysis_prompts()
        file_analyses = []
        for file_data in files_data:
            if file_data['size'] > 50000:
                continue
            try:
                analysis_chain = LLMChain(
                    llm=self.llm,
                    prompt=prompts['file_analysis']
                )
                analysis = analysis_chain.run(
                    file_path=file_data['path'],
                    file_content=file_data['content'][:8000]
                )
                file_analyses.append(f"**{file_data['path']}**:\n{analysis}\n")
                logger.info(f"Analyzed file: {file_data['path']}")
            except Exception as e:
                logger.error(f"Error analyzing file {file_data['path']}: {e}")
        return file_analyses

    def generate_repo_summary(self, file_analyses: List[str]) -> str:
        prompts = self.create_analysis_prompts()
        try:
            combined = "\n".join(file_analyses)
            if len(combined) > 12000:
                documents = [Document(page_content=f) for f in file_analyses]
                map_chain = LLMChain(
                    llm=self.llm,
                    prompt=PromptTemplate(
                        input_variables=["text"],
                        template="Summarize the key insights from this code analysis:\n{text}"
                    )
                )
                reduce_chain = LLMChain(
                    llm=self.llm,
                    prompt=prompts['repo_summary']
                )
                combine_chain = StuffDocumentsChain(
                    llm_chain=reduce_chain,
                    document_variable_name="file_analyses"
                )
                map_reduce_chain = MapReduceDocumentsChain(
                    llm_chain=map_chain,
                    reduce_documents_chain=combine_chain,
                    document_variable_name="text"
                )
                return map_reduce_chain.run(documents)
            else:
                summary_chain = LLMChain(llm=self.llm, prompt=prompts['repo_summary'])
                return summary_chain.run(file_analyses=combined)
        except Exception as e:
            logger.error(f"Error generating repo summary: {e}")
            return "Summary error"

    def analyze_pr_context(self, pr: Any, repo_summary: str) -> str:
        prompts = self.create_analysis_prompts()
        try:
            pr_chain = LLMChain(
                llm=self.llm,
                prompt=prompts['pr_analysis']
            )
            return pr_chain.run(
                pr_title=pr.title,
                pr_description=pr.body or "No description",
                pr_author=pr.user.login,
                repo_analysis=repo_summary
            )
        except Exception as e:
            logger.error(f"Error analyzing PR context: {e}")
            return "PR context analysis error"

    def _get_file_stats(self, files_data: List[Dict[str, str]]) -> Dict[str, int]:
        stats = {}
        for file_data in files_data:
            ext = file_data['extension'] or 'no_ext'
            stats[ext] = stats.get(ext, 0) + 1
        return stats

    def save_analysis_report(self, analysis: Dict[str, Any]) -> None:
        try:
            with open(f"pr_analysis_report_{analysis['pr_number']}.json", "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info("✅ Report saved")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def print_analysis_summary(self, analysis: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print(f"📊 PR ANALYSIS REPORT")
        print("=" * 80)
        print(f"PR #{analysis['pr_number']}: {analysis['pr_title']}")
        print(f"Author: {analysis['pr_author']} | URL: {analysis['pr_url']}")
        print(f"Files Analyzed: {analysis['files_analyzed']}")
        print("\n🔍 Repository Summary:\n" + analysis['repository_summary'])
        print("\n🧠 PR Context Analysis:\n" + analysis['pr_specific_analysis'])
        print("=" * 80)

    def analyze_pr(self, pr: Any) -> Dict[str, Any]:
        logger.info(f"Analyzing PR #{pr.number}: {pr.title}")
        repo_path = self.clone_pr_repo(pr)
        try:
            files = self.extract_code_files(repo_path)
            if not files:
                logger.warning("⚠️ No relevant code files found.")
                return {}
            file_analyses = self.analyze_files(files)
            repo_summary = self.generate_repo_summary(file_analyses)
            pr_context = self.analyze_pr_context(pr, repo_summary)
            return {
                'pr_number': pr.number,
                'pr_title': pr.title,
                'pr_author': pr.user.login,
                'pr_url': pr.html_url,
                'analysis_date': datetime.now().isoformat(),
                'files_analyzed': len(files),
                'repository_summary': repo_summary,
                'pr_specific_analysis': pr_context,
                'file_count_by_type': self._get_file_stats(files)
            }
        finally:
            if repo_path.exists():
                shutil.rmtree(repo_path)

    def run(self) -> None:
        try:
            new_prs = self.get_new_prs()
            if not new_prs:
                logger.info("✅ No new PRs.")
                return
            logger.info(f"Found {len(new_prs)} new PR(s).")
            for pr in new_prs:
                try:
                    result = self.analyze_pr(pr)
                    if result:
                        self.print_analysis_summary(result)
                        self.save_analysis_report(result)
                except Exception as e:
                    logger.error(f"Error analyzing PR #{pr.number}: {e}")
        except Exception as e:
            logger.error(f"Fatal error in run: {e}")


def main():
    try:
        analyzer = EnhancedPRAnalyzer()
        analyzer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
