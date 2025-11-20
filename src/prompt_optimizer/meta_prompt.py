"""
Meta-prompting optimization strategy based on OpenAI's cookbook approach.

This strategy uses a more intelligent model to improve prompts for a less intelligent model,
including scoring and evaluation capabilities using structured generation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class PromptScore(BaseModel):
    """Score card for evaluating prompt quality."""

    clarity: int = Field(ge=1, le=5, description="1-5 score for clarity and structure")
    specificity: int = Field(
        ge=1, le=5, description="1-5 score for specific instructions"
    )
    context: int = Field(
        ge=1, le=5, description="1-5 score for providing adequate context"
    )
    constraints: int = Field(
        ge=1, le=5, description="1-5 score for clear constraints/requirements"
    )
    examples: int = Field(
        ge=1, le=5, description="1-5 score for example usage (if applicable)"
    )
    justification: str = Field(description="Brief explanation of scores")

    @property
    def overall(self) -> float:
        """Calculate overall average score."""
        return (
            self.clarity
            + self.specificity
            + self.context
            + self.constraints
            + self.examples
        ) / 5.0


class MetaPromptOptimizer:
    """
    Meta-prompting optimizer that uses LLMs to improve prompts.

    Based on OpenAI's meta-prompting technique where a more intelligent
    model is used to refine and improve prompts for optimal performance.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        optimizer_model_name: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """
        Initialize the meta-prompt optimizer.

        Args:
            model_name: Model to use for general operations (e.g., gpt-3.5-turbo)
            optimizer_model_name: Optional better model for optimization (e.g., gpt-4)
            temperature: Temperature for generation
        """
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.optimizer_model = (
            ChatOpenAI(model=optimizer_model_name, temperature=temperature)
            if optimizer_model_name
            else self.model
        )
        # Create a structured model for scoring
        self.scoring_model = self.model.with_structured_output(PromptScore)

    def optimize(
        self,
        prompt: str,
        task_description: Optional[str] = None,
        constraints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize a prompt using meta-prompting technique.

        Args:
            prompt: The original prompt to optimize
            task_description: Optional description of what the prompt should achieve
            constraints: Optional constraints the prompt should satisfy

        Returns:
            Dictionary with original prompt, optimized prompt, and scores
        """
        # Build the meta-prompt
        meta_prompt = self._build_meta_prompt(prompt, task_description, constraints)

        # Get the improved prompt
        response = self.optimizer_model.invoke(meta_prompt)
        improved_prompt = response.content.strip()

        # Score both prompts
        original_score = self._score_prompt(prompt)
        improved_score = self._score_prompt(improved_prompt)

        return {
            "original_prompt": prompt,
            "optimized_prompt": improved_prompt,
            "original_score": original_score,
            "improved_score": improved_score,
            "improvement_delta": improved_score.overall - original_score.overall,
            "improvement_percentage": (
                (
                    (improved_score.overall - original_score.overall)
                    / original_score.overall
                )
                * 100
                if original_score.overall > 0
                else 0
            ),
        }

    def _build_meta_prompt(
        self,
        prompt: str,
        task_description: Optional[str],
        constraints: Optional[List[str]],
    ) -> str:
        """Build the meta-prompt for optimization."""

        meta_prompt = f"""You are an expert prompt engineer following the RISEN framework. Your task is to improve the following prompt to make it more effective.

ORIGINAL PROMPT:
{prompt}

"""

        if task_description:
            meta_prompt += f"""TASK DESCRIPTION:
{task_description}

"""

        if constraints:
            meta_prompt += "CONSTRAINTS:\n"
            for constraint in constraints:
                meta_prompt += f"- {constraint}\n"
            meta_prompt += "\n"

        meta_prompt += """OPTIMIZATION CRITERIA:
Please improve the prompt according to these criteria:
- Clear and unambiguous instructions
- Well-structured format
- Specific output requirements  
- Appropriate context and background
- Examples where helpful
- Constraints and edge cases

INSTRUCTIONS:
1. Analyze the original prompt and identify areas for improvement
2. Create an enhanced version that addresses the optimization criteria
3. Ensure the improved prompt is clear, specific, and actionable
4. Maintain the original intent while improving effectiveness

Return ONLY the improved prompt without any explanation or metadata."""

        return meta_prompt

    def _score_prompt(self, prompt: str) -> PromptScore:
        """
        Score a prompt using structured generation.

        Args:
            prompt: The prompt to score

        Returns:
            PromptScore with evaluation
        """
        scoring_prompt = f"""You are an expert prompt evaluator. Score the following prompt on these criteria:

PROMPT TO EVALUATE:
{prompt}

SCORING CRITERIA (rate each 1-5):
1. Clarity: Is the prompt clear, unambiguous, and easy to understand?
2. Specificity: Does it provide specific instructions and requirements?
3. Context: Does it provide adequate context and background information?
4. Constraints: Are constraints, output format, and requirements well-defined?
5. Examples: Does it include helpful examples or templates where appropriate?

Provide your evaluation with scores for each criterion and a brief justification."""

        return self.scoring_model.invoke(scoring_prompt)

    def compare_prompts(
        self,
        original_prompt: str,
        improved_prompt: str,
        test_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare original and improved prompts side by side.

        Args:
            original_prompt: The original prompt
            improved_prompt: The optimized prompt
            test_input: Optional test input to compare outputs

        Returns:
            Comparison results
        """
        comparison = {}

        if test_input:
            # Format prompts with test input
            original_test = f"{original_prompt}\n\nInput: {test_input}"
            improved_test = f"{improved_prompt}\n\nInput: {test_input}"

            # Get outputs from both
            original_response = self.model.invoke(original_test)
            improved_response = self.model.invoke(improved_test)

            comparison["test_results"] = {
                "input": test_input,
                "original_output": original_response.content,
                "improved_output": improved_response.content,
            }

        return comparison
