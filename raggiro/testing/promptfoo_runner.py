"""Promptfoo integration for testing RAG pipelines."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from raggiro.rag.pipeline import RagPipeline

class PromptfooRunner:
    """Runs evaluation tests using promptfoo."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the promptfoo runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure testing settings
        testing_config = self.config.get("testing", {})
        self.max_concurrency = testing_config.get("max_concurrency", 1)
        
        # Initialize RAG pipeline
        self.pipeline = RagPipeline(self.config)
    
    def _evaluate_prompt(self, prompt: str) -> Dict:
        """Evaluate a single prompt using the RAG pipeline.
        
        Args:
            prompt: User prompt to evaluate
            
        Returns:
            Evaluation result
        """
        result = self.pipeline.query(prompt)
        
        return {
            "prompt": prompt,
            "response": result["response"],
            "success": result["success"],
        }
    
    def run_tests(self, prompt_file: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
        """Run tests using promptfoo.
        
        Args:
            prompt_file: Path to promptfoo prompt set YAML file
            output_dir: Output directory
            
        Returns:
            Test results
        """
        prompt_file = Path(prompt_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if promptfoo is installed
        try:
            subprocess.run(["promptfoo", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                "prompt_file": str(prompt_file),
                "success": False,
                "error": "promptfoo not installed or not found in PATH",
            }
        
        # Check if the prompt file exists
        if not prompt_file.exists():
            return {
                "prompt_file": str(prompt_file),
                "success": False,
                "error": f"Prompt file not found: {prompt_file}",
            }
        
        try:
            # Load prompt file
            with open(prompt_file, "r", encoding="utf-8") as f:
                if prompt_file.suffix.lower() in [".yaml", ".yml"]:
                    config = yaml.safe_load(f)
                elif prompt_file.suffix.lower() == ".json":
                    config = json.load(f)
                else:
                    return {
                        "prompt_file": str(prompt_file),
                        "success": False,
                        "error": f"Unsupported prompt file format: {prompt_file.suffix}",
                    }
            
            # Create a custom provider for our RAG pipeline
            provider_file = output_dir / "rag_provider.js"
            
            with open(provider_file, "w", encoding="utf-8") as f:
                f.write("""
const { execSync } = require('child_process');

module.exports = {
  id: 'rag-pipeline',
  async callApi(prompt, options) {
    // Call Python script to evaluate prompt with RAG pipeline
    const output = execSync(`python -m raggiro.testing.eval_prompt "${prompt}"`, {
      encoding: 'utf-8',
    });
    return JSON.parse(output).response;
  },
};
""")
            
            # Create a Python script to evaluate prompts
            script_file = Path(__file__).parent / "eval_prompt.py"
            
            with open(script_file, "w", encoding="utf-8") as f:
                f.write("""
import json
import sys
from pathlib import Path

from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

def evaluate_prompt(prompt):
    # Load configuration
    config = load_config()
    
    # Initialize RAG pipeline
    pipeline = RagPipeline(config)
    
    # Query the pipeline
    result = pipeline.query(prompt)
    
    return {
        "response": result["response"],
        "success": result["success"],
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"response": "Error: No prompt provided", "success": False}))
        sys.exit(1)
    
    prompt = sys.argv[1]
    result = evaluate_prompt(prompt)
    print(json.dumps(result))
""")
            
            # Create a promptfoo configuration file
            config_file = output_dir / "promptfoo_config.yaml"
            
            # Adapt the existing config for promptfoo
            promptfoo_config = {
                "providers": [{"id": "rag-pipeline"}],
                "providerConfigs": {
                    "rag-pipeline": {
                        "config": {
                            "path": str(provider_file.absolute()),
                        }
                    }
                },
                "prompts": config.get("prompts", []),
                "tests": config.get("tests", []),
                "evaluations": config.get("evaluations", []),
                "sharing": False,
            }
            
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(promptfoo_config, f)
            
            # Run promptfoo
            output_file = output_dir / "results.json"
            
            result = subprocess.run([
                "promptfoo",
                "eval",
                "--config", str(config_file),
                "--output-file", str(output_file),
                "--max-concurrency", str(self.max_concurrency),
                "--no-cache",
            ], check=True, capture_output=True, text=True)
            
            # Load and return the results
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                
                return {
                    "prompt_file": str(prompt_file),
                    "output_file": str(output_file),
                    "success": True,
                    "tests_run": len(results.get("results", [])),
                    "results": results,
                }
            else:
                return {
                    "prompt_file": str(prompt_file),
                    "success": False,
                    "error": "No output file generated",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            
        except Exception as e:
            return {
                "prompt_file": str(prompt_file),
                "success": False,
                "error": str(e),
            }

def run_tests(prompt_file: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
    """Run tests using promptfoo.
    
    Args:
        prompt_file: Path to promptfoo prompt set YAML file
        output_dir: Output directory
        
    Returns:
        Test results
    """
    # Load configuration
    from raggiro.utils.config import load_config
    config = load_config()
    
    # Initialize runner
    runner = PromptfooRunner(config)
    
    # Run tests
    return runner.run_tests(prompt_file, output_dir)