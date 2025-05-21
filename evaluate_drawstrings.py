import json
import os
from typing import Dict, List, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from vector_store import VectorStore
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
        self.vector_store = VectorStore()  # Initialize vector store
        self.IMAGE_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations, specifically focused on drawstring policies.

        Task: Analyze this image of a clothing item and determine if it has drawstrings.

        IMPORTANT: Classify as having drawstrings if there is any reasonable indication of functional or decorative drawstrings.

        A drawstring is any cord, string, or similar feature that can be pulled to adjust the fit of the garment. Common locations for drawstrings include:
        - Hood drawstrings (most common in children's outerwear)
        - Waist drawstrings
        - Neck drawstrings
        - Any adjustable cords/strings
        - Toggle cords
        - Elastic cords with toggles
        - Any hanging or adjustable strings

        Look carefully for:
        1. Visible drawstrings hanging from the garment
        2. Drawstring holes or channels where drawstrings would be threaded
        3. Any adjustable cords or strings that could be used to tighten the garment
        4. Toggle mechanisms that might indicate the presence of drawstrings
        5. Any hanging cords or strings, even if they appear decorative

        RED FLAGS (indicate likely drawstrings):
        - Any visible cords or strings
        - Holes or channels in hoods or waists
        - Toggle mechanisms
        - Elastic cords with toggles
        - Any hanging or adjustable features
        - Decorative strings that could be functional

        Provide your analysis with:
        1. Whether the item has drawstrings (be inclusive - if there's any reasonable indication, classify as having drawstrings)
        2. Your confidence level (0.0 to 1.0)
        3. Brief reasoning for your decision, including any features you observed
        """

        self.TEXT_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations.

        Task: Evaluate if the following product listing is for children's upper body outerwear.

        IMPORTANT: Classify as children's outerwear if there is any reasonable indication it might be for children and is outerwear.

        The item is children's upper body outerwear if ANY of these conditions are met:

        1. It might be for children, indicated by ANY of these:
           - Mentions of children, kids, baby, toddler, or youth
           - Children's sizes (2T, 3T, 4, 5, 6, 7, 8, 10, 12, 14)
           - Descriptions suggesting it's for children
           - Listed in children's clothing categories
           - Mentions of being for babies
           - Any indication it might be for children 14 and under

        2. It might be outerwear, indicated by ANY of these:
           - Described as a hoodie, sweatshirt, sweater, jacket, raincoat, cape, or poncho
           - Described as being worn over other clothing
           - Mentioned as outerwear or outer layer
           - Has features typical of outerwear (hoods, heavy fabric, weather protection)
           - Described as warm or protective clothing

        RED FLAGS (indicate likely children's outerwear):
        - Any mention of children or kids
        - Any children's sizes
        - Any outerwear descriptions
        - Any protective clothing features
        - Any mentions of being for babies or toddlers

        Product Listing:
        {listing}

        Similar Cases (for additional context):
        {similar_cases}

        Provide your analysis with:
        1. Whether it's likely children's upper body outerwear (be inclusive - if there's any reasonable indication, classify as children's outerwear)
        2. Your confidence level (0.0 to 1.0)
        3. Brief reasoning for your decision, including any indicators you found
        """

        self.FINAL_PROMPT = """
        You are a product safety expert specializing in children's clothing safety regulations, specifically focused on drawstring policies.

        Task: Make a final evaluation of whether this product listing violates the children's drawstrings policy.

        IMPORTANT: Classify as a violation if there is any reasonable indication that ALL THREE conditions might be met.

        POLICY RULES:
        The item violates the policy if ALL of these conditions are met:
        1. It is likely intended for children size/age 14 and under
        2. It likely contains drawstrings
        3. It is likely upper body outerwear

        You have received two separate analyses:

        Image Analysis:
        {image_analysis}

        Text Analysis:
        {text_analysis}

        Original Product Listing:
        {listing}

        Similar Cases (for additional context, the classification here of 'out_of_scope' means that the case is NOT a violation, while 'etsy.childrens_drawstrings' means that the case is a violation):
        {similar_cases}

        Please make a final decision by:
        1. Considering both analyses' findings and confidence levels
        2. Reviewing the original product listing for any additional context
        3. Considering the similar cases and their classifications to help inform your decision
        4. Evaluating if there is any reasonable indication that ALL THREE conditions are met
        5. Providing your confidence in the final decision
        6. Explaining your reasoning, especially if you disagree with either analysis or if the similar cases suggest a different outcome

        Remember: 
        - If there is any reasonable indication that ALL THREE conditions are met, classify as a violation
        - When in doubt about any condition, consider the overall context and likelihood
        - It's better to flag a potential violation than to miss one
        - Use the original listing and similar cases to resolve any ambiguities in the analyses
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
            # Retrieve similar cases
            similar_cases = self.get_similar_cases(listing, n_results=3)
            similar_cases_str = "\n\n".join([
                f"Similar Case {i+1}:\n{case['text']}\nClassification: {case['metadata']['classification']}"
                for i, case in enumerate(similar_cases)
            ]) if similar_cases else "None found."
            response = await self.client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are a product safety expert specializing in children's clothing regulations."},
                    {"role": "user", "content": self.TEXT_PROMPT.format(listing=formatted_listing, similar_cases=similar_cases_str)}
                ],
                text_format=TextAnalysis,
                temperature=0.2
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return TextAnalysis(is_childrens_outerwear=False, confidence=0.0, reasoning="Error analyzing text")

    def format_for_similarity_search(self, listing: Dict) -> str:
        """Format a listing for similarity search (category, description, keywords, materials)."""
        return f"Category: {listing.get('category', '')}\nDescription: {listing.get('description', '')}\nKeywords: {', '.join(listing.get('keywords', []))}\nMaterials: {', '.join(listing.get('materials', []))}"

    def get_similar_cases(self, listing: Dict, n_results: int = 3):
        """Retrieve similar cases from the vector store."""
        query = self.format_for_similarity_search(listing)
        
        print(f"Query: {query}")
        return self.vector_store.search_similar_cases(query, n_results=n_results)

    async def make_final_evaluation(self, image_analysis: ImageAnalysis, text_analysis: TextAnalysis, listing: Dict) -> FinalEvaluation:
        """Make a final evaluation based on both analyses, the original listing, and similar cases."""
        try:
            formatted_listing = self.format_listing(listing)
            # Retrieve similar cases
            similar_cases = self.get_similar_cases(listing, n_results=3)
            similar_cases_str = "\n\n".join([
                f"Similar Case {i+1}:\n{case['text']}\nClassification: {case['metadata']['classification']}"
                for i, case in enumerate(similar_cases)
            ]) if similar_cases else "None found."
            
            
            prompt = self.FINAL_PROMPT.format(
                image_analysis=f"Has drawstrings: {image_analysis.has_drawstrings}, Confidence: {image_analysis.confidence:.2f}, Reasoning: {image_analysis.reasoning}",
                text_analysis=f"Is children's outerwear: {text_analysis.is_childrens_outerwear}, Confidence: {text_analysis.confidence:.2f}, Reasoning: {text_analysis.reasoning}",
                listing=formatted_listing,
                similar_cases=similar_cases_str
            )
            
            print(f"Prompt: {prompt}")
            response = await self.client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are a product safety expert specializing in children's clothing regulations."},
                    {"role": "user", "content": prompt}
                ],
                text_format=FinalEvaluation,
                temperature=0.3
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            return FinalEvaluation(is_violation=False, confidence=0.0, reasoning="Error in final evaluation")

    async def analyze_all_images(self, image_urls: List[str]) -> ImageAnalysis:
        """Analyze all images in a listing and combine their results."""
        if not image_urls:
            return ImageAnalysis(has_drawstrings=False, confidence=0.0, reasoning="No images available")
        
        # Analyze all images concurrently
        image_analyses = await asyncio.gather(
            *[self.analyze_image(url) for url in image_urls],
            return_exceptions=True
        )
        
        # Filter out any errors and get valid analyses
        valid_analyses = []
        for analysis in image_analyses:
            if isinstance(analysis, Exception):
                print(f"Error analyzing image: {str(analysis)}")
                continue
            if analysis is None:
                continue
            valid_analyses.append(analysis)
        
        if not valid_analyses:
            return ImageAnalysis(has_drawstrings=False, confidence=0.0, reasoning="No valid image analyses")
        
        # If any image shows drawstrings, consider it a positive
        has_drawstrings = any(analysis.has_drawstrings for analysis in valid_analyses)
        
        # Take the highest confidence score
        max_confidence = max(analysis.confidence for analysis in valid_analyses)
        
        # Combine reasoning from all analyses
        reasoning_parts = []
        for i, analysis in enumerate(valid_analyses, 1):
            if analysis.has_drawstrings:
                reasoning_parts.append(f"Image {i}: {analysis.reasoning}")
        
        if not reasoning_parts:
            reasoning_parts = [f"Image {i}: No drawstrings detected" for i in range(1, len(valid_analyses) + 1)]
        
        combined_reasoning = " | ".join(reasoning_parts)
        
        return ImageAnalysis(
            has_drawstrings=has_drawstrings,
            confidence=max_confidence,
            reasoning=f"Analyzed {len(valid_analyses)} images. {combined_reasoning}"
        )

    async def classify_listing(self, listing: Dict) -> DrawStringEvaluation:
        """Classify a single listing using both image and text analysis."""
        try:
            # Get all image URLs from the listing
            image_urls = listing.get("images", [])
            
            # Run both analyses concurrently
            image_analysis, text_analysis = await asyncio.gather(
                self.analyze_all_images(image_urls),
                self.analyze_text(listing)
            )
            
            # Adjust confidence thresholds based on the type of analysis
            # For image analysis, we want to be more lenient since images might be unclear
            # For text analysis, we can be more strict since text is usually more reliable
            image_threshold = 0.15  # Lower threshold for images
            text_threshold = 0.25   # Higher threshold for text
            
            # If both analyses have very low confidence, we can skip the final evaluation
            if image_analysis.confidence < image_threshold and text_analysis.confidence < text_threshold:
                return DrawStringEvaluation(
                    classification="out_of_scope",
                    reasoning=f"Low confidence in both analyses. Image: {image_analysis.confidence:.2f}, Text: {text_analysis.confidence:.2f}"
                )
            
            # If only one analysis has low confidence, we can still proceed but note it in the reasoning
            low_confidence_warning = ""
            if image_analysis.confidence < image_threshold:
                low_confidence_warning += f"Low confidence in image analysis ({image_analysis.confidence:.2f}). "
            if text_analysis.confidence < text_threshold:
                low_confidence_warning += f"Low confidence in text analysis ({text_analysis.confidence:.2f}). "
            
            # Make final evaluation
            final_eval = await self.make_final_evaluation(image_analysis, text_analysis, listing)
            
            # Combine the low confidence warning with the final evaluation reasoning
            combined_reasoning = f"{low_confidence_warning}{final_eval.reasoning}"
            
            return DrawStringEvaluation(
                classification="etsy.childrens_drawstrings" if final_eval.is_violation else "out_of_scope",
                reasoning=combined_reasoning
            )
            
        except Exception as e:
            print(f"Error classifying listing: {str(e)}")
            return DrawStringEvaluation(
                classification="out_of_scope",
                reasoning=f"Error during classification: {str(e)}"
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
            self.analyze_errors(data, predictions, true_labels)
            
            return precision, recall
            
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return 0.0, 0.0  # Return zero metrics on error

async def main():
    evaluator = DrawstringsEvaluator()
    
    # Debug the collection
    evaluator.vector_store.debug_collection()
    
    # Load the dataset
    data = evaluator.load_data("labeled_dataset.json")
    
    # Evaluate the model
    precision, recall = await evaluator.evaluate(data)
    
    print(f"\nEvaluation Results:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

if __name__ == "__main__":
    asyncio.run(main())