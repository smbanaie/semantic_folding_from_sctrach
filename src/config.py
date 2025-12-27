"""
Configuration management using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenRouter Configuration
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    # Memgraph Configuration
    memgraph_uri: str = Field(
        default="bolt://localhost:7687",
        description="Memgraph connection URI",
    )
    memgraph_user: str = Field(default="", description="Memgraph username")
    memgraph_password: str = Field(default="", description="Memgraph password")

    # Corpus Configuration
    corpus_directory: str = Field(
        default="data/corpus",
        description="Directory containing corpus files",
    )
    corpus_files: List[str] = Field(
        default_factory=lambda: ["data/corpus/sample.txt"],
        description="Specific corpus files to process",
    )
    input_directory: str = Field(
        default="data/input",
        description="Additional input directory",
    )
    output_directory: str = Field(
        default="data/output",
        description="Output directory for results",
    )

    # Model Configuration (using recommended OpenRouter free models)
    analyzer_model: str = Field(
        default="deepseek/deepseek-v3.1:free",
        description="Model for Analyzer agent (complex reasoning)",
    )
    splitter_model: str = Field(
        default="z-ai/glm-4.5-air:free",
        description="Model for Splitter agent (fast, efficient)",
    )
    chunker_model: str = Field(
        default="z-ai/glm-4.5-air:free",
        description="Model for Chunker agent (fast, efficient)",
    )
    extractor_model: str = Field(
        default="openai/gpt-oss-20b:free",
        description="Model for Extractor agent (balanced quality/speed)",
    )
    reviewer_model: str = Field(
        default="deepseek/deepseek-v3.1:free",
        description="Model for Reviewer agent (complex reasoning)",
    )

    # Processing Settings
    chunk_size: int = Field(
        default=1000,
        ge=200,
        le=2000,
        description="Target chunk size in tokens",
    )
    chunk_overlap: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Overlap ratio between chunks (0.0-0.5)",
    )
    max_parallel_extractions: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum parallel extraction requests",
    )
    batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Batch size for processing",
    )

    @field_validator("corpus_files", mode="before")
    @classmethod
    def parse_corpus_files(cls, v):
        """Parse corpus_files from string or list."""
        if isinstance(v, str):
            # Handle JSON-like list string: '["file1.txt", "file2.txt"]'
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Handle comma-separated string: "file1.txt,file2.txt"
                return [f.strip() for f in v.split(",") if f.strip()]
        return v

    def get_corpus_files(self) -> List[Path]:
        """
        Get list of corpus files to process.

        Returns:
            List of Path objects for corpus files.
            Falls back to CORPUS_FILES if directory is empty.
        """
        corpus_dir = Path(self.corpus_directory)

        # Check if directory exists and has .txt files
        if corpus_dir.exists() and corpus_dir.is_dir():
            txt_files = list(corpus_dir.glob("*.txt"))
            if txt_files:
                return sorted(txt_files)

        # Fall back to CORPUS_FILES
        files = []
        for file_path in self.corpus_files:
            path = Path(file_path)
            if path.exists():
                files.append(path)
            else:
                # Try relative to corpus_directory
                alt_path = corpus_dir / path.name
                if alt_path.exists():
                    files.append(alt_path)

        if not files:
            raise FileNotFoundError(
                f"No corpus files found in {corpus_dir} or in CORPUS_FILES. "
                "Please add .txt files to data/corpus/ or configure CORPUS_FILES."
            )

        return files

    def load_corpus(self) -> str:
        """
        Load and concatenate all corpus files.

        Returns:
            Combined text from all corpus files.

        Raises:
            FileNotFoundError: If no corpus files are found.
            UnicodeDecodeError: If file encoding issues occur.
        """
        files = self.get_corpus_files()
        texts = []

        for file_path in files:
            try:
                # Try UTF-8 first, fall back to latin-1
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = file_path.read_text(encoding="latin-1")

                texts.append(content)
            except Exception as e:
                raise RuntimeError(
                    f"Error reading corpus file {file_path}: {e}"
                ) from e

        if not texts:
            raise ValueError("No corpus content loaded from files.")

        return "\n\n".join(texts)

    def get_output_path(self, filename: str = "triples.json") -> Path:
        """
        Get output file path.

        Args:
            filename: Name of the output file.

        Returns:
            Path object for the output file.
        """
        output_dir = Path(self.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """
    Get or create the global configuration instance.

    Returns:
        Config instance.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config

