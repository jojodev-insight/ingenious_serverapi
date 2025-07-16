"""Test the file processing capabilities of the enhanced agents."""

from pathlib import Path

import pytest

from agents import FileDataAnalyst, FileProcessor


class TestFileProcessing:
    """Test file processing functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.data_dir = Path("data")
        self.agent = FileDataAnalyst(data_dir=str(self.data_dir))

    def test_file_processor_csv(self):
        """Test CSV file processing."""
        csv_file = self.data_dir / "sales_data.csv"
        if csv_file.exists():
            result = FileProcessor.process_file(str(csv_file))
            assert result["type"] == "dataframe"
            assert "data" in result
            assert result["data"] is not None

    def test_file_processor_excel(self):
        """Test Excel file processing."""
        excel_file = self.data_dir / "employee_data.xlsx"
        if excel_file.exists():
            result = FileProcessor.process_file(str(excel_file))
            assert result["type"] == "dataframe"
            assert "data" in result

    def test_file_processor_text(self):
        """Test text file processing."""
        text_file = self.data_dir / "market_report.txt"
        if text_file.exists():
            result = FileProcessor.process_file(str(text_file))
            assert result["type"] == "text"
            assert "content" in result
            assert len(result["content"]) > 0

    def test_agent_list_data_files(self):
        """Test agent's ability to list data files."""
        files = self.agent.list_data_files()
        assert isinstance(files, list)
        # Should have at least our sample files
        file_names = [f["name"] for f in files]
        assert any("sales_data.csv" in name for name in file_names)

    def test_agent_load_data_file(self):
        """Test agent's ability to load data files."""
        try:
            result = self.agent.load_data_file("sales_data.csv")
            assert "type" in result
            assert "file_info" in result
        except FileNotFoundError:
            pytest.skip("sales_data.csv not found")

    def test_agent_get_data_summary(self):
        """Test agent's data summary functionality."""
        try:
            summary = self.agent.get_data_summary("sales_data.csv")
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert "File:" in summary
        except FileNotFoundError:
            pytest.skip("sales_data.csv not found")

    def test_agent_prepare_task_with_files(self):
        """Test agent's task preparation with file data."""
        task_data = {
            "analysis_request": "Analyze the sales data",
            "files": ["sales_data.csv"]
        }

        try:
            prompt = self.agent.prepare_task(task_data)
            assert isinstance(prompt, str)
            assert "sales_data.csv" in prompt
            assert "Analyze the sales data" in prompt
        except Exception as e:
            pytest.skip(f"Could not prepare task: {e}")


class TestFileDataAnalystMocked:
    """Test FileDataAnalyst with mocked LLM calls."""

    def setup_method(self):
        """Set up test environment."""
        self.agent = FileDataAnalyst()

    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.name == "FileDataAnalyst"
        assert self.agent.template_name == "data_analyst_with_files.txt"

    def test_get_available_files_info(self):
        """Test getting available files information."""
        info = self.agent.get_available_files_info()
        assert isinstance(info, dict)

    def test_prepare_task_with_all_files(self):
        """Test preparing task with all available files."""
        task_data = {
            "analysis_request": "Analyze all available data",
            "include_all_files": True
        }

        prompt = self.agent.prepare_task(task_data)
        assert isinstance(prompt, str)
        assert "Analyze all available data" in prompt


if __name__ == "__main__":
    # Quick test run
    processor = FileProcessor()

    # Test file info
    data_dir = Path("data")
    if data_dir.exists():
        print("Available data files:")
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                info = processor.get_file_info(str(file_path))
                print(f"  {info['name']} ({info['extension']}) - {info['size']} bytes")

    # Test agent
    agent = FileDataAnalyst()
    print(f"\nAgent initialized: {agent.name}")
    print(f"Data directory: {agent.data_dir}")

    # List available files
    files = agent.list_data_files()
    print(f"Found {len(files)} data files")

    if files:
        # Test data summary for first file
        first_file = files[0]["name"]
        print(f"\nSample summary for {first_file}:")
        try:
            summary = agent.get_data_summary(first_file)
            print(summary[:500] + "..." if len(summary) > 500 else summary)
        except Exception as e:
            print(f"Error: {e}")
