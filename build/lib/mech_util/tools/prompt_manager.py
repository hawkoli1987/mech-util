"""
Prompt Manager: Load and render versioned prompts from external YAML files.

Following rules from .cursor/rules/03_prompts_and_provenance.md:
- No hardcoded prompts in Python
- YAML metadata + Jinja2 templates
- Explicit prompt_id and versioning
- Environment overrides supported
- Prompt provenance logging
"""

import os
from pathlib import Path
from typing import Optional, Any
import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel, Field


class PromptMetadata(BaseModel):
    """Metadata for a prompt template."""
    
    prompt_id: str = Field(..., description="Unique prompt identifier (e.g., 'lme.clarify.v1')")
    version: str = Field(..., description="Semantic version (e.g., '1.0.0')")
    description: str = Field(default="", description="Human-readable description")
    author: Optional[str] = Field(default=None, description="Prompt author/owner")
    output_schema: Optional[str] = Field(default=None, description="Expected output schema name")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")


class PromptTemplate(BaseModel):
    """Complete prompt template with metadata and content."""
    
    metadata: PromptMetadata
    template: str = Field(..., description="Jinja2 template content")


class PromptManager:
    """
    Manages loading and rendering of versioned prompt templates.
    
    Features:
    - Loads prompts from YAML files
    - Renders Jinja2 templates with context
    - Supports environment variable overrides
    - Tracks prompt provenance
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize PromptManager.
        
        Args:
            prompts_dir: Root directory for prompts. If not provided, uses PROMPTS_DIR env var.
        """
        # Priority: 1) explicit prompts_dir parameter, 2) PROMPTS_DIR env var, 3) error
        if prompts_dir is not None:
            # Explicit parameter takes priority
            prompts_dir = Path(prompts_dir)
        else:
            # Fall back to environment variable
            env_override = os.environ.get("PROMPTS_DIR")
            if env_override:
                prompts_dir = Path(env_override)
            else:
                # For mech-util, prompts are in agent repos, not here
                # Require explicit prompts_dir parameter if PROMPTS_DIR env var not set
                raise ValueError(
                    "PromptManager requires PROMPTS_DIR environment variable or explicit prompts_dir parameter. "
                    "Prompts are stored in agent repositories, not in mech-util. "
                    "Set PROMPTS_DIR=/path/to/prompts or pass prompts_dir=Path(...)"
                )
        
        self.prompts_dir = Path(prompts_dir)
        
        # Verify directory exists
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=False,  # Prompts are not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Cache for loaded prompts
        self._cache: dict[str, PromptTemplate] = {}
    
    def load_prompt(self, prompt_id: str, use_cache: bool = True) -> PromptTemplate:
        """
        Load a prompt by its ID.
        
        Args:
            prompt_id: Prompt identifier (e.g., 'lme.clarify.v1')
            use_cache: Use cached prompt if available
        
        Returns:
            PromptTemplate with metadata and template content
        
        Raises:
            FileNotFoundError: Prompt file not found
            ValueError: Invalid prompt YAML structure
        """
        # Check cache
        if use_cache and prompt_id in self._cache:
            return self._cache[prompt_id]
        
        # Resolve prompt file path
        # Convention: prompt_id "lme.clarify.v1" -> agents/lme/clarify.yaml
        parts = prompt_id.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid prompt_id format: {prompt_id}. Expected 'category.name.version'")
        
        # Remove version suffix to get file path
        category = parts[0]
        name = '.'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
        
        # Try multiple search paths
        search_paths = [
            self.prompts_dir / "agents" / category / f"{name}.yaml",
            self.prompts_dir / "base" / f"{name}.yaml",
            self.prompts_dir / category / f"{name}.yaml",
        ]
        
        prompt_file = None
        for path in search_paths:
            if path.exists():
                prompt_file = path
                break
        
        if prompt_file is None:
            raise FileNotFoundError(
                f"Prompt file not found for '{prompt_id}'. Searched: {search_paths}"
            )
        
        # Load YAML
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML structure in {prompt_file}")
        
        # Parse metadata and template
        if 'metadata' not in data:
            raise ValueError(f"Missing 'metadata' section in {prompt_file}")
        if 'template' not in data:
            raise ValueError(f"Missing 'template' section in {prompt_file}")
        
        metadata = PromptMetadata(**data['metadata'])
        template_content = data['template']
        
        # Verify prompt_id matches
        if metadata.prompt_id != prompt_id:
            raise ValueError(
                f"Prompt ID mismatch: requested '{prompt_id}', "
                f"but file contains '{metadata.prompt_id}'"
            )
        
        prompt_template = PromptTemplate(metadata=metadata, template=template_content)
        
        # Cache it
        self._cache[prompt_id] = prompt_template
        
        return prompt_template
    
    def render_prompt(
        self,
        prompt_id: str,
        context: dict[str, Any],
        use_cache: bool = True
    ) -> str:
        """
        Load and render a prompt template with context.
        
        Args:
            prompt_id: Prompt identifier
            context: Variables for template rendering
            use_cache: Use cached prompt if available
        
        Returns:
            Rendered prompt string
        """
        prompt_template = self.load_prompt(prompt_id, use_cache=use_cache)
        
        # Render with Jinja2
        template = self.jinja_env.from_string(prompt_template.template)
        rendered = template.render(**context)
        
        return rendered.strip()
    
    def get_metadata(self, prompt_id: str) -> PromptMetadata:
        """
        Get metadata for a prompt without loading the full template.
        
        Args:
            prompt_id: Prompt identifier
        
        Returns:
            Prompt metadata
        """
        prompt_template = self.load_prompt(prompt_id)
        return prompt_template.metadata
    
    def list_prompts(self, category: Optional[str] = None) -> list[str]:
        """
        List available prompts, optionally filtered by category.
        
        Args:
            category: Filter by category (e.g., 'lme', 'dfm')
        
        Returns:
            List of prompt IDs
        """
        prompt_ids = []
        
        # Search in agents/ and base/
        search_dirs = [
            self.prompts_dir / "agents",
            self.prompts_dir / "base",
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Walk through subdirectories
            for yaml_file in search_dir.rglob("*.yaml"):
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    if data and 'metadata' in data:
                        pid = data['metadata'].get('prompt_id')
                        if pid:
                            # Filter by category if specified
                            if category is None or pid.startswith(f"{category}."):
                                prompt_ids.append(pid)
                except Exception:
                    # Skip invalid files
                    continue
        
        return sorted(prompt_ids)
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()


# Singleton instance for convenience
_default_manager: Optional[PromptManager] = None


def get_prompt_manager(prompts_dir: Optional[Path] = None) -> PromptManager:
    """
    Get or create the default prompt manager instance.
    
    This is a convenience function that maintains a singleton instance
    of PromptManager for easy access throughout the application.
    
    Args:
        prompts_dir: Optional directory containing prompt YAML files.
                    If None, uses default prompts directory.
                    If provided, creates a new instance with that directory.
    
    Returns:
        PromptManager instance
    
    Example:
        >>> pm = get_prompt_manager()
        >>> system, user = pm.render_prompt("lme.clarify.v1", {"intent": "..."})
    """
    global _default_manager
    
    if _default_manager is None or prompts_dir is not None:
        _default_manager = PromptManager(prompts_dir)
    
    return _default_manager

