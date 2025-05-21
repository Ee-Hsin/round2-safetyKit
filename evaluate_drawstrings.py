import json
import os
from typing import Dict, List, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
# Load environment variables
load_dotenv()

class DrawStringEvaluation(BaseModel):
    classification: str #either "etsy.childrens_drawstrings" (which means it violates the policy) or "out_of_scope" (which means it does not violate the policy)
    reasoning: str #a brief explanation focusing on why this decision was made

class ImageAnalysis(BaseModel):
    has_drawstrings: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str

class TextAnalysis(BaseModel):
    is_childrens_outerwear: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str

class FinalEvaluation(BaseModel):
    is_violation: bool
    confidence: float
    reasoning: str

class DrawstringsEvaluator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.IMAGE_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations, specifically focused on drawstring policies.

        Task: Analyze this image of a clothing item and determine if it has drawstrings.

        A drawstring is any cord, string, or similar feature that can be pulled to adjust the fit of the garment. Common locations for drawstrings include:
        - Hood drawstrings
        - Waist drawstrings
        - Neck drawstrings
        - Any adjustable cords/strings

        Look carefully for:
        1. Visible drawstrings hanging from the garment
        2. Drawstring holes or channels where drawstrings would be threaded
        3. Any adjustable cords or strings that could be used to tighten the garment

        Provide your analysis with:
        1. Whether the item has drawstrings
        2. Your confidence level (0.0 to 1.0)
        3. Brief reasoning for your decision
        """

        self.TEXT_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations.

        Task: Evaluate if the following product listing is for children's upper body outerwear.

        The item is children's upper body outerwear if ALL of these conditions are met:
        1. It is intended for children size/age 14 and under (including items intended for babies with no size specified)
        2. It is upper body outerwear (hoodies, sweatshirts, sweaters, jackets, raincoats, capes, ponchos)

        IN SCOPE EXAMPLES:
        - Hoodies for children up to age 14
        - Baby sweaters (even if size not specified)
        - Children's jackets up to age 14
        - Kids' raincoats and ponchos

        OUT OF SCOPE EXAMPLES:
        - Any clothing for sizes/ages over 14
        - Items clearly intended for adults
        - Bottoms (pants, shorts)
        - Dresses
        - Skirts
        - Shirts
        - Hats
        - Non-clothing items
        - Baby Wearing Coats

        Product Listing:
        {listing}

        Provide your analysis with:
        1. Whether it's children's upper body outerwear
        2. Your confidence level (0.0 to 1.0)
        3. Brief reasoning for your decision
        """

        self.FINAL_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations, specifically focused on drawstring policies.

        Task: Make a final evaluation of whether this product listing violates the children's drawstrings policy.

        POLICY RULES:
        The item violates the policy if ALL of these conditions are met:
        1. It is intended for children size/age 14 and under
        2. It contains drawstrings
        3. It is upper body outerwear (hoodies, sweatshirts, sweaters, jackets, raincoats, capes, ponchos)

        You have received two separate analyses:

        Image Analysis:
        {image_analysis}

        Text Analysis:
        {text_analysis}

        Please make a final decision by:
        1. Considering both analyses' findings and confidence levels
        2. Evaluating if ALL three conditions are met
        3. Providing your confidence in the final decision
        4. Explaining your reasoning, especially if you disagree with either analysis

        Remember: You must be confident that ALL THREE conditions are met to classify this as a violation.
        """

    def load_data(self, file_path: str) -> List[Dict]:
        """Load the labeled dataset from JSON file."""
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            return json_data["data"]  # Access the "data" array from the JSON structure

    async def analyze_image(self, image_url: str) -> ImageAnalysis:
        """Analyze an image to detect drawstrings."""
        
        print(f"Analyzing image: {image_url}")
        try:
            response = await self.client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": self.IMAGE_PROMPT},
                    {"role": "user", "content": [
                        {"type": "input_image", "image_url": image_url}
                    ]}
                ],
                text_format=ImageAnalysis,
                temperature=0.2
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return ImageAnalysis(has_drawstrings=False, confidence=0.0, reasoning="Error analyzing image")

    async def analyze_text(self, listing: Dict) -> TextAnalysis:
        """Analyze the listing text to determine if it's children's outerwear."""
        try:
            response = await self.client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are a product safety expert specializing in children's clothing regulations."},
                    {"role": "user", "content": self.TEXT_PROMPT.format(listing=json.dumps(listing, indent=2))}
                ],
                text_format=TextAnalysis,
                temperature=0.2
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return TextAnalysis(is_childrens_outerwear=False, confidence=0.0, reasoning="Error analyzing text")

    async def make_final_evaluation(self, image_analysis: ImageAnalysis, text_analysis: TextAnalysis) -> FinalEvaluation:
        """Make a final evaluation based on both analyses."""
        try:
            prompt = self.FINAL_PROMPT.format(
                image_analysis=f"Has drawstrings: {image_analysis.has_drawstrings}, Confidence: {image_analysis.confidence:.2f}, Reasoning: {image_analysis.reasoning}",
                text_analysis=f"Is children's outerwear: {text_analysis.is_childrens_outerwear}, Confidence: {text_analysis.confidence:.2f}, Reasoning: {text_analysis.reasoning}"
            )
            
            response = await self.client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are a product safety expert specializing in children's clothing regulations."},
                    {"role": "user", "content": prompt}
                ],
                text_format=FinalEvaluation,
                temperature=0.2
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            return FinalEvaluation(is_violation=False, confidence=0.0, reasoning="Error in final evaluation")

    async def classify_listing(self, listing: Dict) -> DrawStringEvaluation:
        """Classify a single listing using both image and text analysis."""
        # Get the first image URL from the listing
        image_url = listing.get("images", [""])[0] if listing.get("images") else ""
        
        # Run both analyses concurrently
        image_analysis, text_analysis = await asyncio.gather(
            self.analyze_image(image_url),
            self.analyze_text(listing)
        )
        
        if image_analysis.has_drawstrings or (not image_analysis.has_drawstrings and image_analysis.confidence < 0.2) and text_analysis.is_childrens_outerwear or (not text_analysis.is_childrens_outerwear and text_analysis.confidence < 0.2): 
            # Make final evaluation
            final_eval = await self.make_final_evaluation(image_analysis, text_analysis)
            
            return DrawStringEvaluation(
                classification="etsy.childrens_drawstrings" if final_eval.is_violation else "out_of_scope",
                reasoning=final_eval.reasoning
            )
            
        # If both analyses return out of scope, we can skip the final evaluation
        return DrawStringEvaluation(
                classification="out_of_scope",
                reasoning=f"Low confidence in analyses. Image: {image_analysis.confidence:.2f}, Text: {text_analysis.confidence:.2f}"
            )

    def calculate_precision(self, predictions, true_labels):
        tp = sum(
            1 for p, t in zip(predictions, true_labels)
            if p == "etsy.childrens_drawstrings" and t == "etsy.childrens_drawstrings"
        )
        fp = sum(
            1 for p, t in zip(predictions, true_labels)
            if p == "etsy.childrens_drawstrings" and t != "etsy.childrens_drawstrings"
        )
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def calculate_recall(self, predictions, true_labels):
        tp = sum(
            1 for p, t in zip(predictions, true_labels)
            if p == "etsy.childrens_drawstrings" and t == "etsy.childrens_drawstrings"
        )
        fn = sum(
            1 for p, t in zip(predictions, true_labels)
            if p != "etsy.childrens_drawstrings" and t == "etsy.childrens_drawstrings"
        )
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def analyze_errors(self, data, predictions, true_labels):
        """Analyze misclassifications to understand error patterns."""
        false_positives = []
        false_negatives = []
        
        for item, pred, true in zip(data, predictions, true_labels):
            if pred == "etsy.childrens_drawstrings" and true != "etsy.childrens_drawstrings":
                false_positives.append(item["reviewInput"])
            elif pred != "etsy.childrens_drawstrings" and true == "etsy.childrens_drawstrings":
                false_negatives.append(item["reviewInput"])
        
        print("\nError Analysis:")
        print(f"\nFalse Positives (predicted violation but wasn't): {len(false_positives)}")
        for item in false_positives[:5]:  # Show first 5 examples
            print(f"- {json.dumps(item, indent=2)}")
        
        print(f"\nFalse Negatives (missed violations): {len(false_negatives)}")
        for item in false_negatives[:5]:  # Show first 5 examples
            print(f"- {json.dumps(item, indent=2)}")

    async def evaluate(self, data: List[Dict]) -> Tuple[float, float]:
        """Evaluate the model's performance on the dataset concurrently."""
        # Create tasks for all listings
        tasks = [self.classify_listing(item["reviewInput"]) for item in data]
        
        try:
            # Execute all tasks concurrently and gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out any exceptions and None results
            predictions = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"Task failed with error: {str(result)}")
                    predictions.append("out_of_scope")  # Default to out_of_scope on error
                    continue
                if result is None:
                    print("Task returned None")
                    predictions.append("out_of_scope")  # Default to out_of_scope on None
                    continue
                predictions.append(result.classification)
            
            # Get true labels
            true_labels = [item["expectedOutcome"] for item in data]
            
            # Calculate metrics
            precision = self.calculate_precision(predictions, true_labels)
            recall = self.calculate_recall(predictions, true_labels)
            
            # Analyze errors
            # self.analyze_errors(data, predictions, true_labels)
            
            return precision, recall
            
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return 0.0, 0.0  # Return zero metrics on error

async def main():
    evaluator = DrawstringsEvaluator()
    
    # Load the dataset
    data = evaluator.load_data("labeled_dataset.json")
    
    # Evaluate the model
    precision, recall = await evaluator.evaluate(data)
    
    print(f"\nEvaluation Results:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

if __name__ == "__main__":
    asyncio.run(main())