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

        IMPORTANT: Only classify as having drawstrings if you can clearly identify a functional drawstring or strong evidence of one.

        A drawstring is a functional cord, string, or similar feature that can be pulled to adjust the fit of the garment. Common locations for drawstrings include:
        - Hood drawstrings (most common in children's outerwear)
        - Waist drawstrings
        - Neck drawstrings
        - Any adjustable cords/strings that serve a functional purpose

        Look carefully for:
        1. Visible drawstrings hanging from the garment
        2. Drawstring holes or channels where drawstrings would be threaded
        3. Any adjustable cords or strings that could be used to tighten the garment
        4. Toggle mechanisms that are clearly functional
        5. Any hanging cords or strings that serve a clear adjustment purpose

        RED FLAGS (indicate likely drawstrings):
        - Functional drawstrings in hoods or waists
        - Clear drawstring channels with visible strings
        - Functional toggle mechanisms
        - Adjustable cords that clearly serve a purpose

        NOT CONSIDERED DRAWSTRINGS:
        - Purely decorative strings or cords
        - Non-functional toggles
        - Zipper pulls
        - Button loops
        - Decorative tassels
        - Non-adjustable cords

        Provide your analysis with:
        1. Whether the item has functional drawstrings (be specific about what you see)
        2. Your confidence level (0.0 to 1.0)
        3. Brief reasoning for your decision, including any specific features you observed
        """

        self.TEXT_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations.

        Task: Evaluate if the following product listing is for children's upper body outerwear.

        IMPORTANT: Only classify as children's outerwear if there is clear evidence of both children's sizing AND outerwear type.

        The item is children's upper body outerwear if BOTH of these conditions are met:

        1. It is clearly for children, indicated by ANY of these:
           - Explicitly stated to be for children size/age 14 and under
           - Listed in specific children's sizes (2T, 3T, 4, 5, 6, 7, 8, 10, 12, 14)
           - Clearly described as being for babies, toddlers, or youth
           - Listed in a children's clothing category with clear age/size indicators

        2. It is clearly upper body outerwear, indicated by ANY of these:
           - Explicitly described as a hoodie, sweatshirt, sweater, jacket, raincoat, cape, or poncho
           - Clearly described as being worn over other clothing
           - Specifically mentioned as outerwear or outer layer
           - Has features typical of outerwear (hoods, heavy fabric, weather protection)

        RED FLAGS (indicate likely children's outerwear):
        - Clear children's size indicators (2T-14)
        - Explicit mentions of being for children/babies
        - Clear outerwear descriptions
        - Specific outerwear features mentioned

        NOT CONSIDERED CHILDREN'S OUTERWEAR:
        - Items that only mention "kids" in the title but are clearly for adults
        - Items that don't specify age/size
        - Items that could be for any age
        - Items that don't clearly indicate outerwear type
        - Items that are clearly for adults but mention "kids" in passing

        Product Listing:
        {listing}

        Provide your analysis with:
        1. Whether it's clearly children's upper body outerwear (be specific about the evidence)
        2. Your confidence level (0.0 to 1.0)
        3. Brief reasoning for your decision, including specific indicators you found
        """

        self.FINAL_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations, specifically focused on drawstring policies.

        Task: Make a final evaluation of whether this product listing violates the children's drawstrings policy.

        IMPORTANT: Only classify as a violation if there is clear evidence of ALL THREE conditions.

        POLICY RULES:
        The item violates the policy if ALL of these conditions are met:
        1. It is clearly intended for children size/age 14 and under
        2. It clearly contains functional drawstrings
        3. It is clearly upper body outerwear

        You have received two separate analyses:

        Image Analysis:
        {image_analysis}

        Text Analysis:
        {text_analysis}

        Original Product Listing:
        {listing}

        Please make a final decision by:
        1. Considering both analyses' findings and confidence levels
        2. Reviewing the original product listing for any additional context
        3. Evaluating if ALL THREE conditions are clearly met
        4. Providing your confidence in the final decision
        5. Explaining your reasoning, especially if you disagree with either analysis

        Remember: 
        - You must have clear evidence of ALL THREE conditions to classify as a violation
        - If any condition is unclear or ambiguous, classify as out_of_scope
        - Focus on clear, explicit evidence rather than assumptions
        - Use the original listing to resolve any ambiguities in the analyses
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

    def format_listing(self, listing: Dict) -> str:
        """Format a listing into a clear, structured string highlighting relevant information."""
        formatted = []
        
        # Title and basic info
        if listing.get("title"):
            formatted.append(f"Title: {listing['title']}")
        
        # Description
        if listing.get("description"):
            formatted.append(f"\nDescription: {listing['description']}")
        
        # Size and age information
        size_info = []
        if listing.get("size"):
            size_info.append(f"Size: {listing['size']}")
        if listing.get("age"):
            size_info.append(f"Age: {listing['age']}")
        if size_info:
            formatted.append(f"\nSize/Age Information: {' | '.join(size_info)}")
        
        # Category and tags
        category_info = []
        if listing.get("category"):
            category_info.append(f"Category: {listing['category']}")
        if listing.get("tags"):
            category_info.append(f"Tags: {', '.join(listing['tags'])}")
        if category_info:
            formatted.append(f"\nCategory Information: {' | '.join(category_info)}")
        
        # Material and features
        feature_info = []
        if listing.get("materials"):
            feature_info.append(f"Materials: {', '.join(listing['materials'])}")
        if listing.get("features"):
            feature_info.append(f"Features: {', '.join(listing['features'])}")
        if feature_info:
            formatted.append(f"\nProduct Features: {' | '.join(feature_info)}")
        
        # Images
        if listing.get("images"):
            formatted.append(f"\nNumber of Images: {len(listing['images'])}")
            formatted.append(f"First Image URL: {listing['images'][0]}")
        
        return "\n".join(formatted)

    async def analyze_text(self, listing: Dict) -> TextAnalysis:
        """Analyze the listing text to determine if it's children's outerwear."""
        try:
            formatted_listing = self.format_listing(listing)
            response = await self.client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are a product safety expert specializing in children's clothing regulations."},
                    {"role": "user", "content": self.TEXT_PROMPT.format(listing=formatted_listing)}
                ],
                text_format=TextAnalysis,
                temperature=0.2
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return TextAnalysis(is_childrens_outerwear=False, confidence=0.0, reasoning="Error analyzing text")

    async def make_final_evaluation(self, image_analysis: ImageAnalysis, text_analysis: TextAnalysis, listing: Dict) -> FinalEvaluation:
        """Make a final evaluation based on both analyses and the original listing."""
        try:
            formatted_listing = self.format_listing(listing)
            prompt = self.FINAL_PROMPT.format(
                image_analysis=f"Has drawstrings: {image_analysis.has_drawstrings}, Confidence: {image_analysis.confidence:.2f}, Reasoning: {image_analysis.reasoning}",
                text_analysis=f"Is children's outerwear: {text_analysis.is_childrens_outerwear}, Confidence: {text_analysis.confidence:.2f}, Reasoning: {text_analysis.reasoning}",
                listing=formatted_listing
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
        
        # If either analysis has low confidence, we can skip the final evaluation
        if image_analysis.confidence < 0.2 or text_analysis.confidence < 0.2:
            return DrawStringEvaluation(
                classification="out_of_scope",
                reasoning=f"Low confidence in analyses. Image: {image_analysis.confidence:.2f}, Text: {text_analysis.confidence:.2f}"
            )
        
        # Make final evaluation
        final_eval = await self.make_final_evaluation(image_analysis, text_analysis, listing)
        
        return DrawStringEvaluation(
            classification="etsy.childrens_drawstrings" if final_eval.is_violation else "out_of_scope",
            reasoning=final_eval.reasoning
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