import os
import re
import json
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from scripts.sandbox import Sandbox
from src.rag import RAGPipeline
from src.prompts import MASTER_SYSTEM_PROMPT, ERROR_CORRECTION_PROMPT, VALIDATION_CORRECTION_PROMPT, VALIDATION_PROMPT

AVAILABLE_MODELS = {
    "Gemini 3.1 Flash-Lite (Default)": "gemini-3.1-flash-lite",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.1 Pro (Preview)": "gemini-3.1-pro-preview"
}
DEFAULT_MODEL = "gemini-3.1-flash-lite"

_rag_singleton = None

VALIDATION_FAILURE_PREFIX = "Validation Failed:"

def is_infrastructure_error(stderr: str) -> bool:
    if not stderr:
        return False
    err_lower = stderr.lower()

    # Validation-only failures (code ran fine but didn't match the request)
    # are not infrastructure errors - they're handled by a dedicated
    # correction path, not the RAG/execution-error path.
    if stderr.strip().startswith(VALIDATION_FAILURE_PREFIX):
        return False
    
    # Sandbox errors, PermissionError, Security Shield errors
    if "permissionerror" in err_lower:
        return True
    if "sandbox" in err_lower:
        return True
    if "security shield" in err_lower:
        return True
        
    # ModuleNotFoundError or standard library import failures
    if "modulenotfounderror" in err_lower:
        # Check standard libraries
        std_libs = ["urllib", "http", "socket", "builtins", "subprocess", "ctypes", "sys", "os", "shutil", "tempfile", "time", "datetime", "logging", "multiprocessing", "pydoc", "importlib"]
        if any(lib in err_lower for lib in std_libs):
            return True
        # If it's a ModuleNotFoundError for something that is not a data science module, treat it as infrastructure
        allowed_ds = ["pandas", "plotly", "sklearn", "numpy", "pyarrow", "scipy", "statsmodels", "matplotlib", "seaborn", "openpyxl", "xlrd"]
        match = re.search(r"no module named ['\"]([^'\"]+)['\"]", err_lower)
        if match:
            pkg = match.group(1).split('.')[0]
            if pkg not in allowed_ds:
                return True
        else:
            return True
            
    return False

def get_columns_from_file(file_path: str) -> list:
    import pandas as pd
    import sqlite3
    try:
        df = None
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, nrows=2)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, nrows=2)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path, nrows=2)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
            return df.columns.tolist()
        elif file_path.endswith(".tsv"):
            df = pd.read_csv(file_path, sep="\t", nrows=2)
        elif file_path.endswith(".sqlite"):
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            if tables:
                df = pd.read_sql(f"SELECT * FROM {tables[0]} LIMIT 2", conn)
            conn.close()
        else:
            df = pd.read_csv(file_path, nrows=2)
            
        if df is not None:
            if not df.columns.is_unique:
                new_cols = []
                col_counts = {}
                for col in df.columns:
                    if col in col_counts:
                        col_counts[col] += 1
                        new_cols.append(f"{col}_{col_counts[col]}")
                    else:
                        col_counts[col] = 0
                        new_cols.append(col)
                return new_cols
            return df.columns.tolist()
    except Exception:
        pass
    return []

class DataScienceAgent:
    def __init__(self, api_key: str = None, model_name: str = None, user_id: str = "default_user", rag_pipeline = None):
        if not api_key:
            raise ValueError("Please enter your Gemini API Key before running an analysis.")
        model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=api_key
        )
        self.user_id = re.sub(r'[^A-Za-z0-9_-]', '', user_id.strip()) if user_id else "default_user"
        if not self.user_id:
            self.user_id = "default_user"
        
        if rag_pipeline is not None:
            self.rag = rag_pipeline
        else:
            global _rag_singleton
            if _rag_singleton is None:
                _rag_singleton = RAGPipeline()
            self.rag = _rag_singleton
            
        self.max_retries = 3
        self.memory: List[Dict[str, str]] = []
        self.memory_file = None

    def load_memory(self, project_name: str):
        project_name = re.sub(r'[^A-Za-z0-9_-]', '', project_name.strip())
        if not project_name:
            project_name = "default_project"
        self.memory_file = os.path.join(os.path.abspath("workspace"), self.user_id, project_name, "conversation.json")
        
        # Verify path remains under workspace/user_id to prevent traversal
        expected_prefix = os.path.abspath(os.path.join("workspace", self.user_id))
        if not os.path.abspath(self.memory_file).startswith(expected_prefix):
            raise ValueError("Path traversal attempt detected in memory loading!")

        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
            except Exception:
                self.memory = []
        else:
            self.memory = []

    def save_memory(self):
        if self.memory_file:
            # Keep only last 4 interactions to manage token size (Task 14)
            if len(self.memory) > 4:
                self.memory = self.memory[-4:]
            try:
                with open(self.memory_file, "w", encoding="utf-8") as f:
                    json.dump(self.memory, f, indent=2)
            except Exception:
                pass

    def extract_code(self, text: str) -> str:
        """Extracts Python code from a markdown string, allowing flexible whitespace."""
        pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if len(matches) > 1:
            print(f"Warning: Multiple python code blocks ({len(matches)}) found in LLM response. Using the last one.")
        if matches:
            return matches[-1].strip()
        return ""

    def _get_text(self, content) -> str:
        """Normalize AIMessage.content into a plain string.
        Gemini 3 models can return content as a list of blocks
        (e.g. thinking + text parts) instead of a plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content)

    def truncate_content(self, text: str, max_chars: int = 1500) -> str:
        # Strip/replace python code blocks with a placeholder to avoid breaking code fences and save tokens
        cleaned_text = re.sub(r'```python\s*\n.*?```', '[code block omitted from history]', text, flags=re.IGNORECASE | re.DOTALL)
        if len(cleaned_text) > max_chars:
            return cleaned_text[:max_chars] + "\n...[truncated for memory]..."
        return cleaned_text

    def process_request(self, project_name: str, data_summary: str, dataset_path: str, user_question: str) -> Dict[str, Any]:
        project_name = re.sub(r'[^A-Za-z0-9_-]', '', project_name.strip())
        if not project_name:
            project_name = "default_project"
        self.load_memory(project_name)
        sandbox = Sandbox(project_name=project_name, base_dir=os.path.join("workspace", self.user_id))
        
        history_str = ""
        for msg in self.memory:
            history_str += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
            
        cols = get_columns_from_file(dataset_path)
        cols_formatted = "[\n" + ",\n".join(cols) + "\n]"
            
        prompt = PromptTemplate(
                    template="""
                {system_prompt}

                Available DataFrame Columns:
                {columns_list}

                Data Summary:
                {data_summary}

                Dataset Path: {dataset_path}
                Always load the dataset using this exact path.

                Conversation History:
                {history}

                User Request:
                {user_question}
                """,
                    input_variables=[
                        "system_prompt",
                        "columns_list",
                        "data_summary",
                        "dataset_path",
                        "history",
                        "user_question",
                    ],
                )
        
        chain = prompt | self.llm
        # 1. initial generation
        initial_response = self._get_text(chain.invoke({
            "system_prompt": MASTER_SYSTEM_PROMPT,
            "columns_list": cols_formatted,
            "data_summary": data_summary,
            "dataset_path": dataset_path,
            "history": history_str,
            "user_question": user_question
        }).content)

        current_code = self.extract_code(initial_response)
        
        if not current_code:
            self.memory.append({"role": "user", "content": user_question})
            self.memory.append({"role": "assistant", "content": self.truncate_content(initial_response)})
            self.save_memory()
            return {
                "final_text": initial_response,
                "current_code": "",
                "execution_result": {"success": False, "stdout": "", "stderr": "No Python code generated.", "files_generated": [], "execution_time": 0},
                "retries": 0,
                "success": False
            }

        retries = 0
        success = False
        execution_result = {}
        
        while retries <= self.max_retries and not success:
            execution_result = sandbox.execute_code(current_code)
            is_validation_failure = False
            
            if execution_result["success"]:
                val_prompt = PromptTemplate(
                    template=VALIDATION_PROMPT,
                    input_variables=["user_question", "code", "stdout", "files"]
                )
                val_chain = val_prompt | self.llm
                validation_raw = self._get_text(val_chain.invoke({        # ← change here
                    "user_question": user_question,
                    "code": current_code,
                    "stdout": execution_result["stdout"][:1000],
                    "files": str(execution_result["files_generated"])
                }).content)
                
                try:
                    # Clean up possible markdown fences
                    clean_val = validation_raw.strip().strip("```json").strip("```").strip()
                    val_json = json.loads(clean_val)
                    if val_json.get("status") == "SUCCESS":
                        success = True
                        break
                    else:
                        execution_result["stderr"] = f"{VALIDATION_FAILURE_PREFIX} {val_json.get('reason')}"
                        execution_result["success"] = False
                        is_validation_failure = True
                except json.JSONDecodeError:
                    # Fallback if LLM fails JSON format
                    if "SUCCESS" in validation_raw.upper():
                        success = True
                        break
                    else:
                        execution_result["stderr"] = f"Validation Parsing Failed. Raw output: {validation_raw}"
                        execution_result["success"] = False
                
            if retries == self.max_retries:
                break
                
            error_message = execution_result["stderr"]
            if is_infrastructure_error(error_message):
                break

            if is_validation_failure:
                # The code ran without error but didn't fulfill the request.
                # This is a semantic/intent mismatch, not a technical bug, so
                # skip the RAG doc lookup (pandas/plotly error docs aren't
                # relevant here) and use a correction prompt that realigns
                # the code with the literal user request instead.
                validation_correction_prompt = PromptTemplate(
                    template=VALIDATION_CORRECTION_PROMPT,
                    input_variables=["user_question", "validation_reason", "current_code"]
                )
                validation_correction_chain = validation_correction_prompt | self.llm
                correction_response = self._get_text(validation_correction_chain.invoke({
                    "user_question": user_question,
                    "validation_reason": error_message[len(VALIDATION_FAILURE_PREFIX):].strip()[-1500:],
                    "current_code": current_code
                }).content)
            else:
                retrieved_docs = self.rag.retrieve(error_message)

                correction_prompt = PromptTemplate(
                    template=ERROR_CORRECTION_PROMPT,
                    input_variables=["error_message", "retrieved_docs", "current_code"]
                )

                correction_chain = correction_prompt | self.llm
                correction_response = self._get_text(correction_chain.invoke({   # ← change here
                    "error_message": error_message[-1500:],
                    "retrieved_docs": retrieved_docs,
                    "current_code": current_code
                }).content)
            
            new_code = self.extract_code(correction_response)
            if new_code:
                current_code = new_code
            else:
                current_code = correction_response.strip()
                
            retries += 1

        if retries > 0 and success:
            final_text = re.sub(
                r'```python\n.*?\n```',
                lambda m: f'```python\n{current_code}\n```',   # ← lambda instead of f-string
                initial_response,
                flags=re.DOTALL
            )
        else:
            final_text = initial_response

        self.memory.append({"role": "user", "content": user_question})
        self.memory.append({"role": "assistant", "content": self.truncate_content(final_text)})
        self.save_memory()

        return {
            "final_text": final_text,
            "current_code": current_code,
            "execution_result": execution_result,
            "retries": retries,
            "success": success
        }





