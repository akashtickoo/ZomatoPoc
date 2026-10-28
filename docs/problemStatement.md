# Problem Statement

## AI-Powered Restaurant Recommendation System (Zomato Use Case)

Build an AI-powered restaurant recommendation service inspired by Zomato. The system suggests restaurants that match a user's preferences by combining structured restaurant data with a Large Language Model (LLM).

## Objective

Design and implement an application that:

- Takes user preferences such as location, budget, cuisine and ratings
- Uses a real-world dataset of restaurants
- Uses an LLM to generate personalized, human-like recommendations
- Shows clear, useful results to the user

## System Workflow

### 1. Data Ingestion

- Load and preprocess the Zomato dataset from Hugging Face: [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- Extract the relevant fields, such as restaurant name, location, cuisine, cost and rating.

### 2. User Input

Collect these user preferences:

- **Location** (e.g. Delhi, Bangalore)
- **Budget** (low, medium, high)
- **Cuisine** (e.g. Italian, Chinese)
- **Minimum rating**
- **Additional preferences** (e.g. family-friendly, quick service)

### 3. Integration Layer

- Filter the restaurant data and prepare the subset that matches the user's input.
- Pass the structured results into an LLM prompt.
- Design the prompt so the LLM can reason about the options and rank them.

### 4. Recommendation Engine

Use the LLM to:

- Rank restaurants
- Explain why each recommendation fits
- Summarize the choices (optional)

### 5. Output Display

Show the top recommendations in a user-friendly format. Each recommendation includes:

- Restaurant name
- Cuisine
- Rating
- Estimated cost
- AI-generated explanation
