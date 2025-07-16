"""Data Analyst agent with file processing capabilities."""

from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, ModelConfig


class FileDataAnalyst(BaseAgent):
    """Data analyst agent that can process and analyze various file formats."""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        data_dir: Optional[str] = None
    ):
        """Initialize the File Data Analyst agent.
        
        Args:
            provider: LLM provider to use.
            model_name: Specific model to use.
            model_config: Custom model configuration.
            data_dir: Directory containing data files.
        """
        super().__init__(
            name="FileDataAnalyst",
            template_name="data_analyst_with_files.txt",
            provider=provider,
            model_name=model_name,
            model_config=model_config,
            system_message="You are an expert data analyst. Analyze the provided data directly and generate comprehensive insights.",
            data_dir=data_dir
        )
    
    def prepare_task(self, task_data: Dict[str, Any]) -> str:
        """Prepare the task prompt with actual data content.
        
        Args:
            task_data: Dictionary containing:
                - analysis_request: The analysis request or question
                - files: List of filenames to analyze (optional)
                - include_all_files: Boolean to include all available files (optional)
                - file_pattern: Pattern to match files (optional, e.g., "*.csv")
                - data: Direct data to analyze (optional, overrides files)
        
        Returns:
            Prepared prompt string with actual data content.
        """
        analysis_request = task_data.get("analysis_request", "Please analyze the provided data.")
        
        # Check if data is provided directly
        if "data" in task_data:
            file_summaries = {"Direct Data": str(task_data["data"])}
        else:
            # Load data from files
            files_to_analyze = task_data.get("files", [])
            include_all_files = task_data.get("include_all_files", False)
            file_pattern = task_data.get("file_pattern", "*")
            
            # Determine which files to analyze
            if include_all_files:
                available_files = self.list_data_files(file_pattern)
                files_to_analyze = [f["name"] for f in available_files if f.get("exists", False)]
            elif not files_to_analyze:
                # Default to common file types if no files specified
                available_files = self.list_data_files("*.{csv,xlsx,txt,pdf}")
                files_to_analyze = [f["name"] for f in available_files if f.get("exists", False)][:3]
            
            # Load actual data content instead of just summaries
            file_summaries = {}
            for filename in files_to_analyze:
                try:
                    file_data = self.load_data_file(filename)
                    
                    if file_data.get("type") == "dataframe":
                        df = file_data.get("data")
                        if hasattr(df, 'to_string'):
                            # Include actual data content
                            data_content = f"File: {filename}\n"
                            data_content += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                            data_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                            data_content += f"Data Types:\n{df.dtypes.to_string()}\n\n"
                            data_content += f"Complete Dataset:\n{df.to_string()}\n\n"
                            data_content += f"Statistical Summary:\n{df.describe().to_string()}\n"
                            file_summaries[filename] = data_content
                        elif isinstance(df, dict):
                            # Handle multiple sheets
                            data_content = f"File: {filename} (Multiple Sheets)\n"
                            for sheet_name, sheet_df in df.items():
                                data_content += f"\nSheet: {sheet_name}\n"
                                data_content += f"Shape: {sheet_df.shape[0]} rows × {sheet_df.shape[1]} columns\n"
                                data_content += f"Columns: {', '.join(sheet_df.columns.tolist())}\n"
                                data_content += f"Data:\n{sheet_df.to_string()}\n\n"
                            file_summaries[filename] = data_content
                        else:
                            file_summaries[filename] = f"Data: {str(df)}"
                    
                    elif file_data.get("type") == "text":
                        content = file_data.get("content", "")
                        file_summaries[filename] = f"File: {filename}\nContent:\n{content}"
                    
                    else:
                        file_summaries[filename] = f"File: {filename}\nError: Unsupported file type"
                        
                except Exception as e:
                    file_summaries[filename] = f"Error loading {filename}: {str(e)}"
        
        # Render the prompt with actual data
        return self.render_prompt(
            analysis_request=analysis_request,
            file_summaries=file_summaries
        )
    
    def analyze_data_directly(self, data: Any, analysis_request: str) -> Dict[str, Any]:
        """Analyze data directly without file involvement.
        
        Args:
            data: The data to analyze (DataFrame, dict, string, etc.).
            analysis_request: The analysis request or question.
            
        Returns:
            Analysis results.
        """
        task_data = {
            "analysis_request": analysis_request,
            "data": data
        }
        return self.execute(task_data)
    
    def analyze_sales_data(self, filename: str = "sales_data.csv") -> Dict[str, Any]:
        """Convenience method to analyze sales data.
        
        Args:
            filename: Name of the sales data file.
            
        Returns:
            Analysis results.
        """
        task_data = {
            "analysis_request": "Analyze the sales data and provide insights on revenue trends, product performance, and recommendations for improvement.",
            "files": [filename]
        }
        return self.execute(task_data)
    
    def analyze_employee_data(self, filename: str = "employee_data.xlsx") -> Dict[str, Any]:
        """Convenience method to analyze employee data.
        
        Args:
            filename: Name of the employee data file.
            
        Returns:
            Analysis results.
        """
        task_data = {
            "analysis_request": "Analyze the employee data to identify salary patterns, performance correlations, and department insights.",
            "files": [filename]
        }
        return self.execute(task_data)
    
    def compare_multiple_datasets(self, filenames: List[str]) -> Dict[str, Any]:
        """Compare and analyze multiple datasets.
        
        Args:
            filenames: List of filenames to compare.
            
        Returns:
            Comparative analysis results.
        """
        task_data = {
            "analysis_request": "Compare and analyze the provided datasets, identifying relationships, trends, and insights across all files.",
            "files": filenames
        }
        return self.execute(task_data)
    
    def get_available_files_info(self) -> Dict[str, Any]:
        """Get information about all available data files.
        
        Returns:
            Dictionary containing file information and summaries.
        """
        files_info = self.list_data_files()
        detailed_info = {}
        
        for file_info in files_info:
            filename = file_info["name"]
            try:
                summary = self.get_data_summary(filename)
                detailed_info[filename] = {
                    "file_info": file_info,
                    "summary": summary
                }
            except Exception as e:
                detailed_info[filename] = {
                    "file_info": file_info,
                    "error": str(e)
                }
        
        return detailed_info
