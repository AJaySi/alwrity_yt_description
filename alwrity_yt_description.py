import time #Iwish
import os
import json
import requests
import streamlit as st
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import google.generativeai as genai


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Alwrity - AI YouTube Description Generator (Beta)",
        layout="wide",
    )
    # Remove the extra spaces from margin top.
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
    st.markdown(f"""
      <style>
      [class="st-emotion-cache-7ym5gk ef3psqc12"]{{
            display: inline-block;
            padding: 5px 20px;
            background-color: #4681f4;
            color: #FBFFFF;
            width: 300px;
            height: 35px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            border-radius: 8px;’
      }}
      </style>
    """
    , unsafe_allow_html=True)

    # Hide top header line
    hide_decoration_bar_style = '<style>header {visibility: hidden;}</style>'
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    # Hide footer
    hide_streamlit_footer = '<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>'
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        keywords = st.text_input('**Describe Your YT video Keywords (comma-separated)**', 
                                help="Enter keywords separated by commas.").split(',')
        target_audience = st.multiselect('**Select Your Target Audience**', 
                    ['Beginners', 'Marketers', 'Gamers', 'Foodies', 'Entrepreneurs', 'Students', 'Parents', 
                     'Tech Enthusiasts', 'General Audience', 'News Readers', 'Finance Enthusiasts'], 
                    help="Select the target audience for your video.")
    
    with col2:
        tone_style = st.selectbox('**Select Tone and Style of YT Description**', 
                    ['Casual', 'Professional', 'Humorous', 'Formal', 'Informal', 'Inspirational'],
                    help="Select the tone and style of your video.")
        language = st.selectbox('**Select YT description Language**', 
                    ['English', 'Spanish', 'Chinese', 'Hindi', 'Arabic'], 
                    help="Select the language for the video description.")
    
    
    if st.button('**Generate YT Description**'):
        with st.spinner():
            if not keywords:
                st.error("🚫 Please provide all required inputs.")
            else:
                response = generate_youtube_description(keywords, target_audience, tone_style, language)
                if response:
                    st.subheader(f'**🧕👩: Your Final youtube Description !**')
                    st.write(response)
                    st.write("\n\n\n\n\n\n")
                else:
                    st.error("💥**Failed to write YT Description. Please try again!**")


def generate_youtube_description(keywords, target_audience, tone_style, language):
    """ Generate youtube script generator """

    prompt = f"""
    Please write a descriptive YouTube description in {language} for a video about {keywords} based on the following information:

    Keywords: {', '.join(keywords)}

    Target Audience: {', '.join(target_audience)}

    Language for description: {', '.join(language)}

    Tone and Style: {tone_style}

    Specific Instructions:

    - Include Primary Keywords Early: Place the most important keywords at the beginning to enhance SEO.
    - Write a Compelling Hook: Start with an engaging sentence to grab attention and entice viewers to watch the video.
    - Provide a Brief Overview: Summarize the video's content and what viewers can expect to learn or experience.
    - Use Relevant Keywords: Integrate additional keywords naturally to improve searchability.
    - Add Timestamps: Include timestamps for different sections of the video, if applicable.
    - Include Links: Add links to related videos, playlists, or external resources.
    - Encourage Engagement: Ask viewers to like, comment, and subscribe, and include a clear call to action.
    - Provide Contact Information: Include relevant social media handles, website links, or contact information.
    - Use Clear and Concise Language: Avoid jargon and keep sentences straightforward and easy to understand.
    - Include Hashtags: Use relevant hashtags to increase discoverability, placing them at the end of the description.
    - Tailor the Language and Tone: Adjust to suit the target audience.
    - Engage and Describe: Use descriptive language to make the video sound interesting.
    - Be Concise but Informative: Provide enough context about the video.
    - Highlight Unique Details: Mention any important details or highlights that make the video unique.
    - Ensure Proper Grammar and Spelling: Maintain a high standard of writing.

    Generate a detailed YouTube description that adheres to the above guidelines and includes a compelling hook, a brief overview, relevant keywords, a call to action, hashtags, and any other relevant information. Ensure proper formatting and a clear structure.
    """
    
    try:
        response = generate_text_with_exception_handling(prompt)
        return response
    except Exception as err:
        st.error(f"Exit: Failed to get response from LLM: {err}")
        exit(1)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text_with_exception_handling(prompt):
    """
    Generates text using the Gemini model with exception handling.

    Args:
        api_key (str): Your Google Generative AI API key.
        prompt (str): The prompt for text generation.

    Returns:
        str: The generated text.
    """

    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        generation_config = {
            "temperature": 1,
            "top_p": 0.7,
            "top_k": 0,
            "max_output_tokens": 8192,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text

    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    main()
