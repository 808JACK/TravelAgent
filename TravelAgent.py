from textwrap import dedent
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.groq import Groq
import streamlit as st
import re
import os
from dotenv import load_dotenv
st.title("TripGrok ✈️🧳")
st.caption("Your AI-powered travel companion")

# API keys
load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
serp_api_key = os.environ.get("SERP_API_KEY")

# Function to clean up unwanted tags like <think>
def clean_output(text):
    # Remove <think> sections and any other unwanted tags
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

# Sidebar for selecting options
st.sidebar.title("Travel Options")
option = st.sidebar.radio("Choose an option:", ("Planner", "Train"))

# Initialize agents
if groq_api_key and serp_api_key:

    research = Agent(
        name="plane_agent",
        role="Searches for travel destinations, activities, and accommodations",
        model=Groq("deepseek-r1-distill-llama-70b", api_key=groq_api_key),
        description=dedent(
            """
            You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
            generate a list of search terms for finding relevant travel activities and accommodations.
            Then search the web for each term, analyze the results, and return the 10 most relevant results.
            """
        ),
        instructions=[
            "Generate 3 search terms for the destination and days.",
            "Search Google for each term and return the top 10 results.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )

    planner = Agent(
        name="planner",
        role="Generates a draft itinerary based on user preferences and research results",
        model=Groq("deepseek-r1-distill-llama-70b", api_key=groq_api_key),
        description=dedent(
            """
            You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
            your goal is to generate a draft itinerary that meets the user's needs and preferences.
            """
        ),
        instructions=[
            "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
            "Ensure the itinerary is well-structured, informative, and engaging.",
            "Provide a balanced mix of history, culture, nature, and modern attractions.",
            "Focus on clarity and quality; avoid including internal reasoning or thought processes in the output.",
            "Never make up facts or plagiarize. Always provide proper attribution if quoting specific facts.",
        ],
        add_datetime_to_instructions=True,
    )

    train = Agent(
        name="train_agent",
        role="Searches for train options from starting point to destination",
        model=Groq("deepseek-r1-distill-llama-70b", api_key=groq_api_key),
        description=dedent(
            """
            You are an expert in train travel logistics. Given a starting point and a travel destination,
            search for available train options, including schedules, durations, and booking information.
            """
        ),
        instructions=[
            "Given a starting point and a travel destination, generate a list of 3 search terms related to train travel between these locations.",
            "For each search term, `search_google` and analyze the results.",
            "Return a concise summary of the top 3 train options, including departure times, travel duration, and booking sources.",
            "Ensure the information is accurate and relevant to the user's travel dates if provided.",
            "If no specific travel date is provided, assume the trip starts within the next 7 days from today (March 09, 2025).",
            "Avoid including internal reasoning or commentary in the output.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )

    preferences = st.multiselect(
        "what are your travel preferences?",
        ["Adventure","Relaxation","Culture","Food","History","Nature"],
        default=["Culture","History"]
    )
    # Input fields
    starting_point = st.text_input("Where are you travelling from?")
    destination = st.text_input("Where do you want to go?")
    num_days = st.number_input("How many days do you want to travel for?", min_value=1, max_value=30, value=7)

    if option == "Planner" and st.button("Generate Itinerary"):
        with st.spinner("Processing..."):
            research_result = research.run(f"{destination} for {num_days} days", stream=False)
            itinerary_response = planner.run(
                f"{destination} for {num_days} days with preference {','.join(preferences)} and research results :{research_result.content}",
                stream=False
            )
            cleaned_itinerary = clean_output(itinerary_response.content)
            st.subheader("Draft Itinerary")
            st.write(cleaned_itinerary)

    elif option == "Train" and st.button("Generate Train Options"):
        with st.spinner("Processing..."):
            train_response = train.run(f"Train from {starting_point} to {destination}", stream=False)
            cleaned_train_options = clean_output(train_response.content)
            st.subheader("Train Options")
            st.write(cleaned_train_options)