# content_generator.py
"""
Content Generation Module

This module handles the actual generation of captions and images using the prompts
created by the RL system.

- Caption generation: OpenAI GPT-4o-mini
- Image generation: Gemini 2.5 Flash Image Preview
"""
import os
import json
import base64
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from db import SupabaseImageManager
try:
    # Try to use the newer google.genai package
    import google.genai as genai
    USE_NEW_PACKAGE = True
except ImportError:
    # Fall back to deprecated package
    import google.generativeai as genai
    USE_NEW_PACKAGE = False

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class ContentGenerator:
    """Handles caption and image generation using AI models."""

    def _create_logo_placement_prompt(self, content: str, position: str) -> str:
        """Create AI prompt for logo placement"""
        position_descriptions = {
            "top_left": "top-left corner",
            "top_right": "top-right corner",
            "bottom_left": "bottom-left corner",
            "bottom_right": "bottom-right corner"
        }
        position_desc = position_descriptions.get(position, "bottom-right corner")

        return f"""Add the logo from the second image to the first image at the {position_desc}.

CRITICAL BACKGROUND REMOVAL INSTRUCTIONS:
- Extract ONLY the logo elements from the second image
- REMOVE ALL WHITE BACKGROUNDS from the logo
- REMOVE ALL COLORED BACKGROUNDS from the logo
- REMOVE ALL SOLID BACKGROUNDS from the logo
- The logo must be COMPLETELY TRANSPARENT except for the actual logo elements
- NO white rectangles, circles, or shapes should be visible
- NO background colors should remain
- The logo should have ZERO background

STRICT PLACEMENT INSTRUCTIONS:
- Place the logo at the {position_desc} of the first image
- Do NOT add any text, effects, or creative elements
- Do NOT modify the original image content
- Do NOT add shadows, borders, or styling
- Do NOT add any background to the logo
- Just add the transparent logo, nothing else

MANDATORY: The logo must be placed as a transparent overlay with absolutely no background color or shape."""

    def __init__(self):
        """Initialize the content generator with API clients."""
        self.openai_client = None
        self.gemini_client = None
        self.image_manager = None

        if OPENAI_API_KEY:
            self.openai_client = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=0.7
            ) 

        if GEMINI_API_KEY:
            if USE_NEW_PACKAGE:
                # New google.genai package
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                # Legacy google.generativeai package
                genai.configure(api_key=GEMINI_API_KEY)
                try:
                    # Use the requested model for image generation
                    self.gemini_client = genai.GenerativeModel('gemini-2.5-flash-image-preview')
                except Exception as e:
                    print(f"Warning: Could not initialize gemini-2.5-flash-image-preview ({e}), trying fallback")
                    try:
                        self.gemini_client = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
                    except Exception:
                        print("Warning: Could not initialize any Gemini image generation model")
                        self.gemini_client = None
        else:
            self.gemini_client = None

        # Initialize Supabase image manager
        try:
            self.image_manager = SupabaseImageManager()
        except Exception as e:
            print(f"Warning: Could not initialize Supabase image manager: {e}")
            self.image_manager = None

    def generate_caption(self, caption_prompt: str) -> str:
        """
        Generate a caption using OpenAI GPT-4o-mini.

        Args:
            caption_prompt: The prompt to generate the caption from

        Returns:
            Plain text caption with hashtags
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            response = self.openai_client.invoke([
                HumanMessage(content=caption_prompt)
            ])

            # Return the response content directly (should include hashtags as per prompt)
            return response.content.strip()

        except Exception as e:
            raise RuntimeError(f"Failed to generate caption: {e}")

    def generate_image(self, image_prompt: str) -> str:
        """
        Generate an image using Gemini.

        Args:
            image_prompt: The prompt to generate the image from

        Returns:
            Base64 encoded image data URL or placeholder URL
        """
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured")

        # Warn if {{CAPTION}} placeholder is found (should be replaced before calling this)
        if "{{CAPTION}}" in image_prompt:
            print("⚠️  WARNING: {{CAPTION}} placeholder found in image prompt. Use generate_content() instead of generate_image() directly.")

        try:
            if USE_NEW_PACKAGE:
                # New google.genai package API
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash-image-preview',
                    contents=image_prompt
                )

                # Handle new package response format
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        image_data = None
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data is not None:
                                inline_data = part.inline_data
                                if hasattr(inline_data, 'data') and inline_data.data is not None:
                                    image_data = inline_data.data
                                    break  # Found the image data, exit the loop
                                elif hasattr(inline_data, 'blob') and inline_data.blob is not None:
                                    # Try 'blob' attribute instead
                                    image_data = inline_data.blob
                                    break  # Found the image data, exit the loop

                        # Now check if we found image data
                        if image_data is not None:
                            # Upload to Supabase storage and get public URL
                            try:
                                if self.image_manager:
                                    public_url = self.image_manager.save_image(
                                        image_data=image_data,
                                        filename="generated_image.png",
                                        folder="generated"
                                    )
                                    return public_url
                                else:
                                    raise RuntimeError("Image manager not initialized")
                            except Exception as upload_error:
                                print(f"Failed to upload to Supabase: {upload_error}")
                                # Fallback to base64 data URL
                                base64_image = base64.b64encode(image_data).decode('utf-8')
                                return f"data:image/png;base64,{base64_image}"
                # If no image generated, return a placeholder
                return f"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

            else:
                # Legacy google.generativeai package API
                response = self.gemini_client.generate_content(image_prompt)

                # Get the generated image data
                if response.candidates and response.candidates[0].content.parts:
                    image_part = response.candidates[0].content.parts[0]

                    if hasattr(image_part, 'inline_data'):
                        # Get image data
                        image_data = image_part.inline_data.data

                        # Upload to Supabase storage and get public URL
                        try:
                            if self.image_manager:
                                public_url = self.image_manager.save_image(
                                    image_data=image_data,
                                    filename="generated_image.png",
                                    folder="generated"
                                )
                                return public_url
                            else:
                                raise RuntimeError("Image manager not initialized")
                        except Exception as upload_error:
                            print(f"Failed to upload to Supabase: {upload_error}")
                            # Fallback to base64 data URL
                            mime_type = image_part.inline_data.mime_type
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                            return f"data:{mime_type};base64,{base64_image}"

                raise RuntimeError("No image generated in response")

        except Exception as e:
            # Raise clear error instead of returning placeholder
            error_msg = f"Image generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def generate_image_with_logo(self, image_prompt: str, logo_url: str, logo_position: str = "bottom_right") -> str:
        """
        Generate an image with logo overlay using Gemini API.

        Args:
            image_prompt: The prompt to generate the base image from
            logo_url: URL of the logo image to overlay
            logo_position: Position for logo placement (bottom_right, top_left, etc.)

        Returns:
            Base64 encoded image data URL or public URL
        """
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured")

        try:
            # Create the combined prompt for image generation with logo
            logo_placement_prompt = self._create_logo_placement_prompt(image_prompt, logo_position)

            # For Gemini, we need to provide both images (base image content + logo)
            # The prompt should instruct to generate the base image and then overlay the logo
            combined_prompt = f"""{image_prompt}

            After generating the above image, overlay the logo from this URL: {logo_url}

            {logo_placement_prompt}"""

            # Use the existing generate_image method with the combined prompt
            # Note: This assumes Gemini can handle logo URLs in prompts
            # For more reliable logo placement, we might need to fetch and provide both images
            return self.generate_image(combined_prompt)

        except Exception as e:
            error_msg = f"Image generation with logo failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def generate_content(self, caption_prompt: str, image_prompt: str, business_context: dict = None) -> Dict[str, Any]:
        """
        Generate both caption and image from their respective prompts.

        Args:
            caption_prompt: Prompt for caption generation
            image_prompt: Prompt for image generation
            business_context: Business profile data dictionary for structured context formatting

        Returns:
            Dict with 'caption' and 'image_url' keys
        """
        try:
            caption = self.generate_caption(caption_prompt)

            # Replace {{CAPTION}} placeholder in image_prompt with actual generated caption
            if caption and "{{CAPTION}}" in image_prompt:
                image_prompt = image_prompt.replace("{{CAPTION}}", caption)
                print(f"🔄 Replaced {{CAPTION}} placeholder in image prompt")

            # Format and append structured business context to image prompt if provided
            if business_context:
                formatted_context = format_business_context(business_context)
                image_prompt += f"\n\nBusiness Context: \n\n{formatted_context}"
                print(f"📋 Appended structured business context to image prompt")

            # Check if logo should be added to the image
            logo_url = business_context.get("logo_url") if business_context else None
            if logo_url:
                image_url = self.generate_image_with_logo(image_prompt, logo_url)
                print(f"🏷️ Generated image with logo overlay")
            else:
                image_url = self.generate_image(image_prompt)

            return {
                "caption": caption,
                "image_url": image_url,
                "status": "success"
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }


def format_business_context(profile_data: dict) -> str:
    """
    Format business profile data into structured context for image generation LLMs.

    Args:
        profile_data: Business profile dictionary from database

    Returns:
        Formatted business context string
    """
    context_parts = []

    # Core business information
    if profile_data.get("business_name"):
        context_parts.append(f"Business Name: {profile_data['business_name']}")

    if profile_data.get("industries"):
        industries = ", ".join(profile_data["industries"]) if isinstance(profile_data["industries"], list) else profile_data["industries"]
        context_parts.append(f"Industry: {industries}")

    if profile_data.get("business_types"):
        business_types = ", ".join(profile_data["business_types"]) if isinstance(profile_data["business_types"], list) else profile_data["business_types"]
        context_parts.append(f"Business Type: {business_types}")

    # Description and value proposition
    if profile_data.get("business_description"):
        context_parts.append(f"Description: {profile_data['business_description']}")

    if profile_data.get("unique_value_proposition"):
        context_parts.append(f"Unique Value Proposition: {profile_data['unique_value_proposition']}")

    # Brand and audience information
    if profile_data.get("brand_voice"):
        context_parts.append(f"Brand Voice: {profile_data['brand_voice']}")

    if profile_data.get("target_audience"):
        context_parts.append(f"Target Audience: {profile_data['target_audience']}")

    # Visual preferences - Always include color information
    primary_color = profile_data.get("primary_color", "Not specified")
    context_parts.append(f"Primary Color: {primary_color}")

    secondary_color = profile_data.get("secondary_color", "Not specified")
    context_parts.append(f"Secondary Color: {secondary_color}")

    # Join all parts with proper formatting
    return "\n\n".join(context_parts)


# Convenience functions for direct use
def generate_caption(caption_prompt: str) -> str:
    """Generate a caption from a prompt."""
    generator = ContentGenerator()
    return generator.generate_caption(caption_prompt)


def generate_image(image_prompt: str) -> str:
    """Generate an image from a prompt."""
    generator = ContentGenerator()
    return generator.generate_image(image_prompt)


def generate_content(caption_prompt: str, image_prompt: str, business_context: dict = None) -> Dict[str, Any]:
    """Generate both caption and image from their prompts."""
    generator = ContentGenerator()
    return generator.generate_content(caption_prompt, image_prompt, business_context)


if __name__ == "__main__":
    # Example usage
    caption_prompt = "Write an engaging Instagram caption about productivity tips for entrepreneurs, including relevant hashtags."
    image_prompt = "Create a professional image showing a laptop with coffee and notebooks, modern office setting."

    generator = ContentGenerator()

    try:
        result = generator.generate_content(caption_prompt, image_prompt)
        print("Generated Content:")
        print(f"Caption: {result['caption']}")
        print(f"Image URL: {result['image_url'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")